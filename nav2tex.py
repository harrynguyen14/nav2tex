import torch
import torch.nn as nn

from encoder import NaFlexViTEncoder, build_transform
from decoder import DecoderLM


class Nav2Tex(nn.Module):
    def __init__(self, config, freeze_encoder: bool = False):
        super().__init__()
        model_name = getattr(config, "encoder_model", "naflexvit_base_patch16_gap.e300_s576_in1k")
        self.max_patches = getattr(config, "max_patches", 576)
        self.encoder = NaFlexViTEncoder(model_name=model_name, pretrained=True, freeze_backbone=freeze_encoder)
        self.decoder = DecoderLM(config)

    def forward(self, images, input_ids, attention_mask=None, labels=None, true_len=None, encoder_key_mask=None):
        encoder_output = self.encoder(images, max_patches=self.max_patches)
        return self.decoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_output=encoder_output,
            labels=labels,
            true_len=true_len,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(images, max_patches=self.max_patches)

    @torch.no_grad()
    def generate(self, images, tokenizer, max_new_tokens: int = 512, device="cpu", encoder_key_mask=None):
        images = images.to(device)
        encoder_output = self.encoder(images, max_patches=self.max_patches)

        bos = tokenizer.bos_token_id
        input_ids = torch.tensor([[bos]], device=device)

        for _ in range(max_new_tokens):
            logits = self.decoder(input_ids, encoder_output=encoder_output)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if (next_id == tokenizer.eos_token_id).all():
                break

        return input_ids[:, 1:]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)