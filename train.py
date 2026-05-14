import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from safetensors.torch import save_file, load_file
from tqdm import tqdm

from config import get_config
from dataset import build_dataloader
from encoder import build_transform
from evaluate import compute_metrics
from nav2tex import Nav2Tex


def _flatten_tensors(d: dict, prefix: str) -> tuple[dict, dict]:
    tensors, scalars = {}, {}
    for k, v in d.items():
        full_key = f"{prefix}/{k}"
        if isinstance(v, torch.Tensor):
            tensors[full_key] = v.cpu()
        elif isinstance(v, dict):
            t, s = _flatten_tensors(v, full_key)
            tensors.update(t)
            scalars.update(s)
        else:
            scalars[full_key] = v
    return tensors, scalars


def _unflatten_tensors(tensors: dict, scalars: dict, prefix: str) -> dict:
    result = {}
    sub_prefix = prefix + "/"
    for key, val in {**tensors, **scalars}.items():
        if not key.startswith(sub_prefix):
            continue
        parts = key[len(sub_prefix):].split("/")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return result


def _make_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def _make_optimizer_phase1(model: Nav2Tex, config) -> AdamW:
    """Phase 1: encoder frozen, only decoder params."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or (name.endswith(".weight") and param.dim() == 1):
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW(
        [
            {"params": decay,    "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.phase1_lr,
    )


def _make_optimizer_phase2(model: Nav2Tex, config) -> AdamW:
    """Phase 2: differential LR — encoder gets 10x lower LR than decoder."""
    enc_decay, enc_nodecay = [], []
    dec_decay, dec_nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_enc = name.startswith("encoder.")
        is_nodecay = "norm" in name or "bias" in name or (name.endswith(".weight") and param.dim() == 1)
        if is_enc:
            (enc_nodecay if is_nodecay else enc_decay).append(param)
        else:
            (dec_nodecay if is_nodecay else dec_decay).append(param)
    return AdamW(
        [
            {"params": dec_decay,    "lr": config.phase2_lr,     "weight_decay": config.weight_decay},
            {"params": dec_nodecay,  "lr": config.phase2_lr,     "weight_decay": 0.0},
            {"params": enc_decay,    "lr": config.phase2_enc_lr, "weight_decay": config.weight_decay},
            {"params": enc_nodecay,  "lr": config.phase2_enc_lr, "weight_decay": 0.0},
        ],
        lr=config.phase2_lr,
    )


def _find_latest_checkpoint(save_dir: Path, phase: int) -> Path | None:
    pattern = f"phase{phase}_step_*"
    ckpts = sorted(save_dir.glob(pattern), key=lambda p: int(p.name.split("_")[-1]))
    return ckpts[-1] if ckpts else None


def _save_checkpoint(model, optimizer, scheduler, config, phase: int, step: int, loss: float, save_dir: Path):
    ckpt_dir = save_dir / f"phase{phase}_step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_file({k: v.cpu() for k, v in raw_model.state_dict().items()}, ckpt_dir / "model.safetensors")

    opt_tensors, opt_scalars = _flatten_tensors(optimizer.state_dict(), "optimizer")
    sch_tensors, sch_scalars = _flatten_tensors({"state": scheduler.state_dict()}, "scheduler")
    trainer_tensors = {**opt_tensors, **sch_tensors}
    trainer_scalars = {**opt_scalars, **sch_scalars, "step": step, "phase": phase, "loss": loss}
    metadata = {k: json.dumps(v) for k, v in trainer_scalars.items()}
    if not trainer_tensors:
        trainer_tensors["_sentinel"] = torch.zeros(1)
    save_file(trainer_tensors, ckpt_dir / "trainer.safetensors", metadata=metadata)

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)


def _load_model_weights(model, ckpt_dir: Path):
    sd = load_file(ckpt_dir / "model.safetensors", device="cpu")
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)


def _load_trainer_state(optimizer, scheduler, ckpt_dir: Path, device) -> int:
    trainer_tensors = load_file(ckpt_dir / "trainer.safetensors", device="cpu")
    trainer_tensors.pop("_sentinel", None)
    from safetensors import safe_open
    with safe_open(ckpt_dir / "trainer.safetensors", framework="pt", device="cpu") as f:
        metadata = f.metadata()
    trainer_scalars = {k: json.loads(v) for k, v in metadata.items()}

    opt_sd = _unflatten_tensors(trainer_tensors, trainer_scalars, "optimizer")
    sch_sd = _unflatten_tensors(trainer_tensors, trainer_scalars, "scheduler")
    sch_sd = sch_sd.get("state", sch_sd)

    for state in opt_sd.get("state", {}).values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    optimizer.load_state_dict(opt_sd)
    scheduler.load_state_dict(sch_sd)
    return int(trainer_scalars["step"])


def _run_phase(
    model: Nav2Tex,
    loader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config,
    phase: int,
    total_steps: int,
    start_step: int,
    device,
    save_dir: Path,
    amp_ctx,
    sdp_backends,
):
    model.train()
    step             = start_step
    data_iter        = iter(loader)
    running_loss     = 0.0
    running_lm_loss  = 0.0
    running_len_loss = 0.0
    running_gnorm    = 0.0
    tokens_seen      = 0
    t0               = time.perf_counter()

    pbar = tqdm(total=total_steps, initial=start_step, desc=f"phase{phase}", dynamic_ncols=True)

    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss     = 0.0
        accum_lm_loss  = 0.0
        accum_len_loss = 0.0
        batch_tokens   = 0

        for i in range(config.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            pixel_values     = batch["pixel_values"].to(device)
            encoder_key_mask = batch["encoder_key_mask"].to(device) if batch["encoder_key_mask"] is not None else None
            input_ids        = batch["input_ids"].to(device)
            labels           = batch["labels"].to(device)
            attention_mask   = batch["attention_mask"].to(device)
            true_len         = batch["true_len"].to(device)
            batch_tokens    += attention_mask.sum().item()

            is_last_accum = (i == config.grad_accum - 1)
            sync_ctx = contextlib.nullcontext() if is_last_accum or not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.no_sync()

            with sync_ctx, amp_ctx:
                _sdp_ctx = (
                    sdpa_kernel(sdp_backends)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with _sdp_ctx:
                    loss, lm_loss, len_loss = model(
                        pixel_values,
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        true_len=true_len,
                        encoder_key_mask=encoder_key_mask,
                    )
                    loss = loss / config.grad_accum

            loss.backward()
            accum_loss     += loss.item()
            accum_lm_loss  += lm_loss.item() / config.grad_accum
            accum_len_loss += len_loss.item() / config.grad_accum

        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm).item()
        optimizer.step()
        scheduler.step()

        step             += 1
        running_loss     += accum_loss
        running_lm_loss  += accum_lm_loss
        running_len_loss += accum_len_loss
        running_gnorm    += gnorm
        tokens_seen      += batch_tokens
        pbar.update(1)

        if step % config.log_every_n_steps == 0:
            elapsed     = time.perf_counter() - t0
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
            n           = config.log_every_n_steps
            avg_loss    = running_loss     / n
            avg_lm      = running_lm_loss  / n
            avg_len     = running_len_loss / n
            avg_gnorm   = running_gnorm    / n
            lr_now      = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}", lm=f"{avg_lm:.4f}", len=f"{avg_len:.4f}",
                gnorm=f"{avg_gnorm:.3f}", lr=f"{lr_now:.2e}", tok_s=f"{tok_per_sec:,.0f}",
            )
            tqdm.write(
                f"[p{phase}] step={step:>7d}  loss={avg_loss:.4f}  lm={avg_lm:.4f}  len={avg_len:.4f}"
                f"  gnorm={avg_gnorm:.3f}  lr={lr_now:.2e}  tok/s={tok_per_sec:,.0f}"
            )
            running_loss = running_lm_loss = running_len_loss = running_gnorm = tokens_seen = 0
            t0 = time.perf_counter()

        if step % config.save_every_n_steps == 0:
            _save_checkpoint(model, optimizer, scheduler, config, phase, step, accum_loss, save_dir)
            tqdm.write(f"  [p{phase}] checkpoint saved at step {step}")

    pbar.close()
    _save_checkpoint(model, optimizer, scheduler, config, phase, step, accum_loss, save_dir)
    return step


@torch.no_grad()
def _run_evaluation(model: Nav2Tex, loader, tokenizer, config, device, phase: int):
    model.train(False)
    hypotheses, references = [], []
    max_samples = getattr(config, "val_samples", 500)

    for batch in tqdm(loader, desc=f"validation p{phase}", dynamic_ncols=True, leave=False):
        if len(hypotheses) >= max_samples:
            break
        pixel_values     = batch["pixel_values"].to(device)
        encoder_key_mask = batch["encoder_key_mask"].to(device) if batch["encoder_key_mask"] is not None else None
        ref_ids          = batch["labels"]

        pred_ids = model.generate(
            pixel_values,
            tokenizer,
            max_new_tokens=config.max_seq_len,
            device=device,
            encoder_key_mask=encoder_key_mask,
        )

        for pred, ref in zip(pred_ids, ref_ids):
            pred_text = tokenizer.decode(pred.tolist(), skip_special_tokens=True)
            ref_text  = tokenizer.decode(
                [t for t in ref.tolist() if t != -100],
                skip_special_tokens=True,
            )
            hypotheses.append(pred_text)
            references.append(ref_text)

    metrics = compute_metrics(hypotheses, references)
    print(
        f"[p{phase}] val  bleu4={metrics['bleu4']:.2f}  "
        f"exact_match={metrics['exact_match']:.2f}%  "
        f"edit_dist={metrics['edit_dist']:.4f}"
    )
    model.train(True)
    return metrics


def train():
    config   = get_config()
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.cuda_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_bf16 = config.bf16 and device.type == "cuda"
    amp_ctx  = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    sdp_backends = (
        [SDPBackend.FLASH_ATTENTION]
        if config.flash_attn
        else [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    )

    print(f"device={device}  bf16={use_bf16}")

    transform     = build_transform(model_name=config.encoder_model, is_training=True)
    val_transform = build_transform(model_name=config.encoder_model, is_training=False)

    val_loader = None
    if getattr(config, "val_data_glob", None):
        val_dict = {**vars(config), "data_glob": config.val_data_glob, "batch_size": config.val_batch_size}
        val_cfg  = argparse.Namespace(**val_dict)
        val_loader = build_dataloader(val_cfg, transform=val_transform, split="val")

    from transformers import NougatTokenizerFast
    tokenizer = NougatTokenizerFast.from_pretrained(config.tokenizer_dir)

    # ── Phase 1: frozen encoder ──────────────────────────────────────────────
    print("\n=== Phase 1: frozen encoder ===")
    loader1   = build_dataloader(config, transform=transform, split="train")

    stats = loader1.dataset.score_stats()
    print(f"dataset: total={stats['total']}  cpe={stats['n_cpe']} ({stats['cpe_pct']}%)")

    model = Nav2Tex(config, freeze_encoder=True).to(device)
    print(f"parameters (trainable): {model.num_parameters() / 1e6:.1f}M")

    steps_per_epoch1 = max(1, len(loader1) // config.grad_accum)
    total_steps1     = steps_per_epoch1 * config.phase1_epochs
    print(f"phase1: steps_per_epoch={steps_per_epoch1}  total={total_steps1}")

    opt1  = _make_optimizer_phase1(model, config)
    sch1  = _make_scheduler(opt1, config.warmup_steps, total_steps1)

    start_step1 = 0
    resume_ckpt = Path(config.resume) if config.resume else _find_latest_checkpoint(save_dir, phase=1)
    if resume_ckpt and resume_ckpt.exists():
        print(f"resuming phase1 from {resume_ckpt}")
        _load_model_weights(model, resume_ckpt)
        start_step1 = _load_trainer_state(opt1, sch1, resume_ckpt, device)
        print(f"resumed at step {start_step1}")

    if config.compile:
        model = torch.compile(model)

    _run_phase(model, loader1, opt1, sch1, config, phase=1,
               total_steps=total_steps1, start_step=start_step1,
               device=device, save_dir=save_dir, amp_ctx=amp_ctx, sdp_backends=sdp_backends)

    if val_loader is not None:
        _run_evaluation(model, val_loader, tokenizer, config, device, phase=1)

    # ── Phase 2: full model ──────────────────────────────────────────────────
    print("\n=== Phase 2: full model (differential LR) ===")
    p2_dict   = {**vars(config), "data_glob": config.phase2_data_glob}
    p2_config = argparse.Namespace(**p2_dict)
    loader2   = build_dataloader(p2_config, transform=transform, split="train")

    stats2 = loader2.dataset.score_stats()
    print(f"dataset: total={stats2['total']}  cpe={stats2['n_cpe']} ({stats2['cpe_pct']}%)")

    # unfreeze encoder
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    print(f"parameters (trainable after unfreeze): {model.num_parameters() / 1e6:.1f}M")

    steps_per_epoch2 = max(1, len(loader2) // config.grad_accum)
    total_steps2     = steps_per_epoch2 * config.phase2_epochs
    print(f"phase2: steps_per_epoch={steps_per_epoch2}  total={total_steps2}")

    opt2  = _make_optimizer_phase2(model, config)
    warmup2 = min(getattr(config, "phase2_warmup_steps", 200), max(1, total_steps2 // 10))
    sch2  = _make_scheduler(opt2, warmup2, total_steps2)

    start_step2  = 0
    resume_ckpt2 = _find_latest_checkpoint(save_dir, phase=2)
    if resume_ckpt2 and resume_ckpt2.exists():
        print(f"resuming phase2 from {resume_ckpt2}")
        _load_model_weights(model, resume_ckpt2)
        start_step2 = _load_trainer_state(opt2, sch2, resume_ckpt2, device)
        print(f"resumed at step {start_step2}")

    _run_phase(model, loader2, opt2, sch2, config, phase=2,
               total_steps=total_steps2, start_step=start_step2,
               device=device, save_dir=save_dir, amp_ctx=amp_ctx, sdp_backends=sdp_backends)

    if val_loader is not None:
        _run_evaluation(model, val_loader, tokenizer, config, device, phase=2)

    print("training complete")


if __name__ == "__main__":
    train()