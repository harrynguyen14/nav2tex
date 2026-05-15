import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from encoder import NaFlexViTEncoder
from decoder import DecoderLM


class Nav2Tex(nn.Module):
    def __init__(self, config, freeze_encoder: bool = False):
        super().__init__()
        model_name = getattr(config, "encoder_model", "naflexvit_base_patch16_gap.e300_s576_in1k")
        self.max_patches  = getattr(config, "max_patches", 576)
        self.encoder = NaFlexViTEncoder(model_name=model_name, pretrained=True, freeze_backbone=freeze_encoder)
        self.decoder = DecoderLM(config)

        if getattr(config, "grad_ckpt", False):
            self.encoder.enable_grad_checkpointing()

    def forward(self, images, input_ids, attention_mask=None, labels=None, true_len=None, encoder_key_mask=None):
        encoder_output = self.encoder(images, max_patches=self.max_patches)
        return self.decoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_output=encoder_output,
            labels=labels,
            true_len=true_len,
            encoder_key_mask=encoder_key_mask,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(images, max_patches=self.max_patches)

    @torch.no_grad()
    def generate(self, images, tokenizer, max_new_tokens: int = 512, device="cpu",
                 encoder_key_mask=None, num_beams: int = 1):
        images = images.to(device)
        encoder_output = self.encoder(images, max_patches=self.max_patches)
        if encoder_key_mask is not None:
            encoder_key_mask = encoder_key_mask.to(device)

        if num_beams <= 1:
            return self._greedy(encoder_output, encoder_key_mask, tokenizer, max_new_tokens, device)
        return self._beam_search(encoder_output, encoder_key_mask, tokenizer, max_new_tokens, device, num_beams)

    def _greedy(self, encoder_output, encoder_key_mask, tokenizer, max_new_tokens, device):
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        past_key_values = None

        for _ in range(max_new_tokens):
            logits, past_key_values = self.decoder.generate_step(
                input_ids, encoder_output, encoder_key_mask, past_key_values=past_key_values
            )
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if (next_id == tokenizer.eos_token_id).all():
                break

        return input_ids[:, 1:]

    def _beam_search(self, encoder_output, encoder_key_mask, tokenizer, max_new_tokens, device, num_beams):
        B     = encoder_output.size(0)
        eos_id = tokenizer.eos_token_id
        bos_id = tokenizer.bos_token_id
        vocab  = self.decoder.vocab_size

        # expand encoder output for beams: (B, S, C) -> (B*K, S, C)
        enc_exp = encoder_output.unsqueeze(1).expand(-1, num_beams, -1, -1).reshape(
            B * num_beams, *encoder_output.shape[1:]
        )
        mask_exp = None
        if encoder_key_mask is not None:
            mask_exp = encoder_key_mask.unsqueeze(1).expand(-1, num_beams, -1).reshape(B * num_beams, -1)

        beam_scores = torch.full((B, num_beams), -math.inf, device=device)
        beam_scores[:, 0] = 0.0
        beam_tokens = torch.full((B, num_beams, 1), bos_id, dtype=torch.long, device=device)
        beam_done   = torch.zeros(B, num_beams, dtype=torch.bool, device=device)
        past_key_values = None

        for step in range(max_new_tokens):
            flat_ids = beam_tokens.reshape(B * num_beams, -1)

            logits, past_key_values = self.decoder.generate_step(
                flat_ids, enc_exp, mask_exp, past_key_values=past_key_values
            )

            log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)  # (B*K, V)
            log_probs = log_probs.view(B, num_beams, vocab)

            if step > 0:
                done_mask = beam_done.unsqueeze(-1).expand_as(log_probs)
                log_probs = log_probs.masked_fill(done_mask, -math.inf)
                log_probs[:, :, eos_id] = log_probs[:, :, eos_id].masked_fill(beam_done, 0.0)

            candidate_scores = (beam_scores.unsqueeze(-1) + log_probs).view(B, num_beams * vocab)
            topk_scores, topk_ids = candidate_scores.topk(num_beams, dim=-1)

            beam_idx  = topk_ids // vocab
            token_idx = topk_ids % vocab

            new_tokens  = beam_tokens[torch.arange(B, device=device).unsqueeze(1), beam_idx]
            beam_tokens = torch.cat([new_tokens, token_idx.unsqueeze(-1)], dim=-1)
            beam_scores = topk_scores
            beam_done   = beam_done[torch.arange(B, device=device).unsqueeze(1), beam_idx]
            beam_done   = beam_done | (token_idx == eos_id)

            # reorder cache using built-in EncoderDecoderCache API
            flat_beam_idx = (torch.arange(B, device=device).unsqueeze(1) * num_beams + beam_idx).reshape(-1)
            past_key_values.reorder_cache(flat_beam_idx)

            if beam_done.all():
                break

        return beam_tokens[:, 0, 1:]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
