import torch
import torch.nn as nn

from encoder import NaFlexViTEncoder
from decoder import DecoderLM


def _stack_past(past_list):
    """Stack a list of per-beam past_key_values into a single batched tuple."""
    # past_list: list of length B*K, each element is a tuple of (num_layers,) tuples of (k, v) tensors
    n_layers = len(past_list[0])
    return tuple(
        (
            torch.cat([p[layer][0] for p in past_list], dim=0),
            torch.cat([p[layer][1] for p in past_list], dim=0),
        )
        for layer in range(n_layers)
    )


def _reorder_past(past, beam_idx):
    """Reorder batched past_key_values by beam_idx, return as list of per-beam tuples."""
    # past: tuple of (k, v) per layer, k/v shape (B*K, heads, seq, head_dim)
    # returns list of length B*K with per-beam past
    B_K = beam_idx.size(0)
    reordered = tuple(
        (past[layer][0][beam_idx], past[layer][1][beam_idx])
        for layer in range(len(past))
    )
    # split back into per-beam list so next step can re-stack after reorder
    return [
        tuple((reordered[layer][0][i:i+1], reordered[layer][1][i:i+1]) for layer in range(len(reordered)))
        for i in range(B_K)
    ]


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
        bos = tokenizer.bos_token_id
        input_ids = torch.tensor([[bos]], device=device)
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
        import math
        B = encoder_output.size(0)
        eos_id = tokenizer.eos_token_id
        bos_id = tokenizer.bos_token_id
        vocab  = self.decoder.vocab_size

        # expand encoder output for beams: (B, S, C) -> (B*num_beams, S, C)
        enc_exp  = encoder_output.unsqueeze(1).expand(-1, num_beams, -1, -1).reshape(B * num_beams, *encoder_output.shape[1:])
        mask_exp = None
        if encoder_key_mask is not None:
            mask_exp = encoder_key_mask.unsqueeze(1).expand(-1, num_beams, -1).reshape(B * num_beams, -1)

        # beam state: scores (B, num_beams), sequences (B, num_beams, T)
        beam_scores = torch.full((B, num_beams), -math.inf, device=device)
        beam_scores[:, 0] = 0.0  # start with first beam active
        beam_tokens = torch.full((B, num_beams, 1), bos_id, dtype=torch.long, device=device)
        beam_done   = torch.zeros(B, num_beams, dtype=torch.bool, device=device)
        past_key_values = [None] * (B * num_beams)

        for step in range(max_new_tokens):
            # flatten beams into batch dim
            flat_ids = beam_tokens.reshape(B * num_beams, -1)

            # run one decode step per beam
            logits_list, pkv_list = [], []
            # batch all beams together if past is uniform (None or all set)
            if all(p is None for p in past_key_values):
                logits, pkv = self.decoder.generate_step(flat_ids, enc_exp, mask_exp, past_key_values=None)
                logits_list = logits
                pkv_list    = pkv  # tuple of layers
            else:
                # stack past_key_values across the batch dimension
                stacked_pkv = _stack_past(past_key_values)
                logits_list, pkv_list = self.decoder.generate_step(flat_ids, enc_exp, mask_exp, past_key_values=stacked_pkv)

            log_probs = torch.log_softmax(logits_list[:, -1, :].float(), dim=-1)  # (B*num_beams, V)
            log_probs = log_probs.view(B, num_beams, vocab)

            # mask finished beams — only EOS gets score, rest -inf
            if step > 0:
                done_mask = beam_done.unsqueeze(-1).expand_as(log_probs)
                log_probs = log_probs.masked_fill(done_mask, -math.inf)
                log_probs[:, :, eos_id] = log_probs[:, :, eos_id].masked_fill(beam_done, 0.0)

            # candidate scores: (B, num_beams, V)
            candidate_scores = beam_scores.unsqueeze(-1) + log_probs
            candidate_scores = candidate_scores.view(B, num_beams * vocab)

            # pick top num_beams
            topk_scores, topk_ids = candidate_scores.topk(num_beams, dim=-1)  # (B, num_beams)
            beam_idx  = topk_ids // vocab   # which beam each came from
            token_idx = topk_ids % vocab    # which token

            # reorder sequences and past
            new_tokens = beam_tokens[torch.arange(B, device=device).unsqueeze(1), beam_idx]  # (B, K, T)
            beam_tokens = torch.cat([new_tokens, token_idx.unsqueeze(-1)], dim=-1)
            beam_scores = topk_scores

            # reorder done flags
            beam_done = beam_done[torch.arange(B, device=device).unsqueeze(1), beam_idx]
            beam_done = beam_done | (token_idx == eos_id)

            # reorder past_key_values
            flat_beam_idx = (torch.arange(B, device=device).unsqueeze(1) * num_beams + beam_idx).reshape(-1)
            past_key_values = _reorder_past(pkv_list, flat_beam_idx)

            if beam_done.all():
                break

        # return best beam (index 0) for each item in batch, strip BOS
        return beam_tokens[:, 0, 1:]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)