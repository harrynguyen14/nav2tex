import sys
sys.path.insert(0, r'D:\Nav2Tex')

import torch
from transformers import NougatTokenizerFast
from nav2tex import Nav2Tex
import argparse

# minimal config
cfg = argparse.Namespace(
    encoder_model="naflexvit_base_patch16_gap.e300_s576_in1k",
    vocab_size=50000,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
    d_model=768,
    n_heads=12,
    n_layers=2,        # shallow for quick test
    d_ff=3072,
    dropout=0.0,
    squeeze_ratio=4,
    lam_lambda=0.1,
    label_smoothing=0.1,
    max_seq_len=32,
)

tok = NougatTokenizerFast.from_pretrained(r'D:\Nav2Tex')
model = Nav2Tex(cfg, freeze_encoder=True)
model.train(False)
print(f"Model params: {model.num_parameters()/1e6:.1f}M")

# --- test forward ---
B, C, H, W = 2, 3, 224, 224
images = torch.randn(B, C, H, W)
latex  = [r"\frac{1}{2}", r"x^2 + y^2 = z^2"]
enc    = tok(latex, return_tensors="pt", padding=True, truncation=True, max_length=31)
input_ids      = torch.cat([torch.full((B,1), tok.bos_token_id), enc["input_ids"]], dim=1)
labels_ids     = torch.cat([enc["input_ids"], torch.full((B,1), tok.eos_token_id)], dim=1)
attention_mask = torch.cat([torch.ones(B,1,dtype=torch.bool), enc["attention_mask"].bool()], dim=1)
labels_ids[labels_ids == tok.pad_token_id] = -100
true_len = attention_mask.sum(dim=1).float() / cfg.max_seq_len

print(f"images: {tuple(images.shape)}")
print(f"input_ids: {tuple(input_ids.shape)}  attn_mask: {tuple(attention_mask.shape)}")

with torch.no_grad():
    loss, lm_loss, len_loss = model(
        images, input_ids,
        attention_mask=attention_mask,
        labels=labels_ids,
        true_len=true_len,
    )
print(f"forward OK  loss={loss.item():.4f}  lm={lm_loss.item():.4f}  len={len_loss.item():.4f}")

# --- test generate (single sample) ---
img1 = images[:1]
out_ids = model.generate(img1, tok, max_new_tokens=20, device="cpu")
print(f"generate output shape: {tuple(out_ids.shape)}")
decoded = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
print(f"decoded: {repr(decoded)}")
print("ALL OK")
