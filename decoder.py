import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos   = torch.arange(max_seq_len).float()
    freqs = torch.outer(pos, theta)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    B, H, T, D = x.shape
    x_c   = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    x_rot = x_c * freqs[:T].unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(x_rot).reshape(B, H, T, D).to(dtype)


class SqueezeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.d_model % config.squeeze_ratio == 0

        self.n_heads   = config.n_heads
        self.head_dim  = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        d_sq = config.d_model // config.squeeze_ratio

        self.q_proj    = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_squeeze = nn.Linear(config.d_model, d_sq,           bias=False)
        self.k_expand  = nn.Linear(d_sq,           config.d_model, bias=False)
        self.v_squeeze = nn.Linear(config.d_model, d_sq,           bias=False)
        self.v_expand  = nn.Linear(d_sq,           config.d_model, bias=False)
        self.out_proj  = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, freqs, attention_mask=None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_expand(F.silu(self.k_squeeze(x))).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_expand(F.silu(self.v_squeeze(x))).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)

        drop = self.dropout_p if self.training else 0.0

        if attention_mask is None:
            # no padding — use is_causal=True so flash attention can run
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            # padding present — build additive mask, flash attn won't run but efficient will
            pad_mask  = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_bias = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype), diagonal=1
            ).unsqueeze(0).expand(B, 1, T, T).clone()
            attn_bias = attn_bias.masked_fill(pad_mask, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=drop)

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads   = config.n_heads
        self.head_dim  = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        self.q_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate     = nn.Parameter(torch.zeros(1))

    def forward(self, x, encoder_output, encoder_key_mask=None):
        B, T, C = x.shape
        S = encoder_output.size(1)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_output).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_output).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if encoder_key_mask is not None:
            # encoder_key_mask: (B, S') — truncate or pad to match actual encoder output length S
            mask = encoder_key_mask[:, :S]
            attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        drop = self.dropout_p if self.training else 0.0
        out  = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        out  = self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))
        return torch.tanh(self.gate) * out


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1     = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2     = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


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
        self.mlp      = nn.Sequential(nn.Linear(config.d_model, config.d_model // 2), nn.GELU(), nn.Linear(config.d_model // 2, 1))
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


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1   = nn.LayerNorm(config.d_model)
        self.sa      = SqueezeAttention(config)
        self.norm2   = nn.LayerNorm(config.d_model)
        self.cross   = CrossAttention(config)
        self.norm3   = nn.LayerNorm(config.d_model)
        self.ffn     = FFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs, attention_mask=None, encoder_output=None, encoder_key_mask=None):
        x = x + self.dropout(self.sa(self.norm1(x), freqs, attention_mask))
        if encoder_output is not None:
            x = x + self.dropout(self.cross(self.norm2(x), encoder_output, encoder_key_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class DecoderLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.embed_drop  = nn.Dropout(config.dropout)
        self.layers      = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.norm_out    = nn.LayerNorm(config.d_model)
        self.lm_head     = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.lam         = LengthAwareModule(config)
        self.enc_norm    = nn.LayerNorm(config.d_model)

        head_dim = config.d_model // config.n_heads
        self.register_buffer("rope_freqs", _build_rope_freqs(head_dim, config.max_seq_len), persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, encoder_output=None, labels=None, true_len=None, encoder_key_mask=None):
        _, T  = input_ids.shape
        freqs = self.rope_freqs[:T]
        x     = self.embed_drop(self.token_embed(input_ids))

        pred_len = None
        if encoder_output is not None:
            encoder_output = self.enc_norm(encoder_output)
            pred_len, len_emb = self.lam(encoder_output)
            x = x + len_emb.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, freqs=freqs, attention_mask=attention_mask, encoder_output=encoder_output, encoder_key_mask=encoder_key_mask)

        logits = self.lm_head(self.norm_out(x))

        if labels is None:
            return logits

        lm_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
            label_smoothing=getattr(self.config, "label_smoothing", 0.1),
        )

        if pred_len is not None and true_len is not None:
            len_loss   = F.smooth_l1_loss(pred_len, true_len)
            lam_lambda = getattr(self.config, "lam_lambda", 0.01)
            return lm_loss + lam_lambda * len_loss, lm_loss, len_loss

        return lm_loss, lm_loss, torch.zeros(1, device=input_ids.device)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)