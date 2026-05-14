"""
Benchmark batch size sweet spot for Nav2Tex on current GPU.
Runs forward+backward at increasing batch sizes, measures throughput and VRAM.
Usage: python benchmark_bs.py
"""
import argparse
import gc
import time
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/workspace/nav2tex")

from nav2tex import Nav2Tex


def make_config(batch_size):
    return argparse.Namespace(
        encoder_model="naflexvit_base_patch16_gap.e300_s576_in1k",
        max_patches=576,
        vocab_size=50000,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        d_model=768,
        n_heads=12,
        n_layers=8,
        d_ff=3072,
        dropout=0.0,
        squeeze_ratio=4,
        lam_lambda=0.1,
        label_smoothing=0.1,
        max_seq_len=256,
    )


def run_batch(model, batch_size, seq_len, device, amp_ctx):
    H = W = 384
    images    = torch.randn(batch_size, 3, H, W, device=device)
    input_ids = torch.randint(2, 50000, (batch_size, seq_len), device=device)
    labels    = torch.randint(2, 50000, (batch_size, seq_len), device=device)
    attn_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    true_len  = torch.rand(batch_size, device=device)

    with amp_ctx:
        loss, _, _ = model(images, input_ids, attention_mask=attn_mask,
                           labels=labels, true_len=true_len)
        loss.backward()


def benchmark(batch_sizes, seq_len=256, warmup=2, iters=5):
    device = torch.device("cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    print(f"{'batch':>6}  {'tok/s':>10}  {'img/s':>8}  {'VRAM GB':>8}  {'status'}")
    print("-" * 55)

    for bs in batch_sizes:
        cfg = make_config(bs)
        try:
            model = Nav2Tex(cfg, freeze_encoder=True).to(device)
            model.train()

            # warmup
            for _ in range(warmup):
                model.zero_grad(set_to_none=True)
                run_batch(model, bs, seq_len, device, amp_ctx)
            torch.cuda.synchronize()

            # measure
            t0 = time.perf_counter()
            for _ in range(iters):
                model.zero_grad(set_to_none=True)
                run_batch(model, bs, seq_len, device, amp_ctx)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            vram = torch.cuda.max_memory_allocated(device) / 1e9
            imgs_per_sec = bs * iters / elapsed
            toks_per_sec = bs * seq_len * iters / elapsed

            print(f"{bs:>6}  {toks_per_sec:>10,.0f}  {imgs_per_sec:>8.1f}  {vram:>8.2f}  OK")

        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6}  {'':>10}  {'':>8}  {'':>8}  OOM")
        except Exception as e:
            print(f"{bs:>6}  {'':>10}  {'':>8}  {'':>8}  ERROR: {e}")
        finally:
            try:
                del model
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)


if __name__ == "__main__":
    batch_sizes = [8, 16, 24, 32, 48, 64, 80, 96, 128]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    benchmark(batch_sizes)
