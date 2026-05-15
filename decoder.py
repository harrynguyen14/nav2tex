import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel


class LengthAwareModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads  = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm     = nn.LayerNorm(config.d_model)
        self.mlp      = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.len_proj = nn.Linear(1, config.d_model)

    def forward(self, encoder_out):
        B, S, C = encoder_out.shape
        q = self.q_proj(encoder_out).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_out).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_out).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = self.out_proj(out.transpose(1, 2).reshape(B, S, C))
        x = self.norm(out + encoder_out).mean(dim=1)
        pred_len = self.mlp(x)
        return pred_len.squeeze(-1), self.len_proj(pred_len)


class _LamConfig:
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads


class DecoderLM(nn.Module):
    PRETRAINED  = "Norm/nougat-latex-base"
    ENCODER_DIM = 768
    DECODER_DIM = 1024

    def __init__(self, config):
        super().__init__()
        self.config = config

        attn_impl = "flash_attention_2" if getattr(config, "flash_attn", False) else "sdpa"
        ved = VisionEncoderDecoderModel.from_pretrained(
            self.PRETRAINED,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16 if getattr(config, "bf16", True) else torch.float32,
        )
        self.mbart = ved.decoder

        self.enc_proj = nn.Linear(self.ENCODER_DIM, self.DECODER_DIM, bias=False)
        self.enc_norm = nn.LayerNorm(self.DECODER_DIM)

        self.lam = LengthAwareModule(_LamConfig(d_model=self.DECODER_DIM, n_heads=16))

        self.label_smoothing = getattr(config, "label_smoothing", 0.1)
        self.lam_lambda      = getattr(config, "lam_lambda", 0.01)
        self.vocab_size      = self.mbart.config.vocab_size

    def _enc_attn_mask(self, encoder_key_mask, S, device):
        if encoder_key_mask is None:
            return None
        return encoder_key_mask[:, :S].to(dtype=torch.long, device=device)

    def forward(self, input_ids, attention_mask=None, encoder_output=None,
                labels=None, true_len=None, encoder_key_mask=None):
        enc = self.enc_norm(self.enc_proj(encoder_output))
        pred_len, len_embed = self.lam(enc)
        enc = enc + len_embed

        enc_attn_mask = self._enc_attn_mask(encoder_key_mask, enc.size(1), enc.device)

        out = self.mbart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_attn_mask,
            use_cache=False,
        )
        logits = out.logits

        if labels is None:
            return logits

        lm_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        len_loss = (
            F.smooth_l1_loss(pred_len, true_len)
            if true_len is not None
            else torch.zeros(1, device=input_ids.device)
        )
        return lm_loss + self.lam_lambda * len_loss, lm_loss, len_loss

    def generate_step(self, input_ids, encoder_output, encoder_key_mask=None, past_key_values=None):
        enc = self.enc_norm(self.enc_proj(encoder_output))
        _, len_embed = self.lam(enc)
        enc = enc + len_embed
        enc_attn_mask = self._enc_attn_mask(encoder_key_mask, enc.size(1), enc.device)
        out = self.mbart(
            input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_attn_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return out.logits, out.past_key_values

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
