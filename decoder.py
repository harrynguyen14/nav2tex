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
        return pred_len.squeeze(-1), self.len_proj(pred_len).unsqueeze(1)


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

        dtype = torch.bfloat16 if getattr(config, "bf16", True) else torch.float32
        ved = VisionEncoderDecoderModel.from_pretrained(
            self.PRETRAINED,
            attn_implementation="eager",
            dtype=dtype,
        )
        # DonutSwinModel encoder does not support flash_attention_2 —
        # extract decoder weights and reload with flash attn impl
        if getattr(config, "flash_attn", False):
            from transformers import MBartForCausalLM
            mbart = MBartForCausalLM._from_config(
                ved.decoder.config,
                attn_implementation="flash_attention_2",
            ).to(dtype=dtype)
            mbart.load_state_dict(ved.decoder.state_dict())
            self.mbart = mbart
        else:
            self.mbart = ved.decoder

        self.enc_proj = nn.Linear(self.ENCODER_DIM, self.DECODER_DIM, bias=False)
        self.enc_norm = nn.LayerNorm(self.DECODER_DIM)

        self.lam = LengthAwareModule(_LamConfig(d_model=self.DECODER_DIM, n_heads=16))

        self.label_smoothing = getattr(config, "label_smoothing", 0.1)
        self.lam_lambda      = getattr(config, "lam_lambda", 1.0)
        self.vocab_size      = self.mbart.config.vocab_size

        if getattr(config, "grad_ckpt", False):
            self.mbart.model.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def _chunked_cross_entropy(self, hidden: torch.Tensor, labels: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        # Project chunk-by-chunk: max live tensor is (chunk, V) not (B*T, V).
        # Gradient flows through hidden -> each chunk_logits normally.
        lm_head = self.mbart.lm_head
        B, T, D = hidden.shape
        flat_h      = hidden.reshape(B * T, D)
        flat_labels = labels.reshape(B * T)
        n_valid     = (flat_labels != -100).sum().clamp(min=1)
        total_loss  = hidden.new_zeros(1)
        for i in range(0, B * T, chunk):
            chunk_h      = flat_h[i : i + chunk]
            chunk_lbl    = flat_labels[i : i + chunk]
            chunk_logits = lm_head(chunk_h)
            loss = F.cross_entropy(
                chunk_logits, chunk_lbl,
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
                reduction="sum",
            )
            total_loss = total_loss + loss
        return total_loss / n_valid

    def _enc_attn_mask(self, encoder_key_mask, S, device):
        if encoder_key_mask is None:
            return None
        mask = encoder_key_mask.to(dtype=torch.long, device=device)
        if mask.size(1) < S:
            pad = torch.zeros(mask.size(0), S - mask.size(1), dtype=torch.long, device=device)
            mask = torch.cat([mask, pad], dim=1)
        return mask[:, :S]

    def forward(self, input_ids, attention_mask=None, encoder_output=None,
                labels=None, true_len=None, encoder_key_mask=None):
        enc = self.enc_norm(self.enc_proj(encoder_output))
        pred_len, len_embed = self.lam(enc)
        enc = enc + len_embed

        enc_attn_mask = self._enc_attn_mask(encoder_key_mask, enc.size(1), enc.device)

        # Capture post-layernorm hidden state via hook on the decoder module itself.
        # This gets the final normalized hidden state (after layer_norm, before lm_head)
        # without materialising the full (B*T, V) logits tensor.
        _last_hidden: list[torch.Tensor] = []
        if labels is not None:
            hook = self.mbart.model.decoder.register_forward_hook(
                lambda _m, _inp, out: _last_hidden.append(out[0])
            )

        out = self.mbart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_attn_mask,
            use_cache=False,
        )

        if labels is None:
            return out.logits

        hook.remove()
        last_hidden = _last_hidden[0]
        del out  # free logits immediately

        lm_loss = self._chunked_cross_entropy(last_hidden, labels)
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
        target_dtype = next(self.mbart.parameters()).dtype
        enc = enc.to(dtype=target_dtype)
        out = self.mbart(
            input_ids=input_ids,
            encoder_hidden_states=enc,
            encoder_attention_mask=None,
            use_cache=False,
        )
        return out.logits, None

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
