import argparse
from pathlib import Path

_DEFAULT_TOKENIZER_DIR = str(Path(__file__).parent)
_DEFAULT_SAVE_DIR      = "/workspace/checkpoints"

# Dataset paths per split
_DATA_RAW   = "/workspace/data/train/raw/*.parquet"
_DATA_LIGHT = "/workspace/data/train/light/*.parquet"
_DATA_HEAVY = "/workspace/data/train/heavy/*.parquet"


def get_config():
    p = argparse.ArgumentParser(description="Train Nav2Tex (encoder+decoder)")

    # tokenizer
    p.add_argument("--tokenizer-dir",  default=_DEFAULT_TOKENIZER_DIR)
    p.add_argument("--pad-token-id",   type=int, default=1)
    p.add_argument("--bos-token-id",   type=int, default=0)
    p.add_argument("--eos-token-id",   type=int, default=2)

    # data
    p.add_argument("--data-glob",         nargs="+", default=[_DATA_RAW, _DATA_LIGHT])
    p.add_argument("--data-glob-ratios",  nargs="+", type=float, default=[0.50, 0.50],
                   help="per-glob sample ratio for phase1 (raw 50%%, light 50%%)")
    p.add_argument("--max-latex-chars",     type=int,   default=1024)
    p.add_argument("--max-seq-len",         type=int,   default=768)
    p.add_argument("--max-patches",         type=int,   default=576)
    p.add_argument("--patch-size",          type=int,   default=16)
    p.add_argument("--cpe-score-threshold", type=int,   default=400)
    p.add_argument("--cpe-ratio",           type=float, default=0.20)

    # encoder
    p.add_argument("--encoder-model", default="naflexvit_base_patch16_gap.e300_s576_in1k")

    # LAM
    p.add_argument("--lam-lambda", type=float, default=1.0)

    # training - phase 1 (frozen encoder; decoder pretrained, aligns enc_proj fast)
    p.add_argument("--phase1-epochs", type=int,   default=2)
    p.add_argument("--phase1-lr",     type=float, default=5e-4)

    # training - phase 2 (full model; encoder needs more adapt time than decoder)
    p.add_argument("--phase2-epochs",    type=int,   default=8)
    p.add_argument("--phase2-lr",        type=float, default=5e-5)
    p.add_argument("--phase2-enc-lr",    type=float, default=5e-6)
    p.add_argument("--phase2-data-glob",        nargs="+", default=[_DATA_RAW, _DATA_LIGHT, _DATA_HEAVY])
    p.add_argument("--phase2-data-glob-ratios", nargs="+", type=float, default=[0.20, 0.40, 1.0],
                   help="per-glob sample ratio for phase2 (raw 20%%, light 40%%, heavy 100%%)")
    p.add_argument("--val-data-glob",    default=None)
    p.add_argument("--val-batch-size",   type=int, default=8)
    p.add_argument("--val-samples",      type=int, default=500)

    # training - shared
    p.add_argument("--batch-size",          type=int,   default=16)
    p.add_argument("--grad-accum",          type=int,   default=8)
    p.add_argument("--warmup-steps",        type=int,   default=200)
    p.add_argument("--phase2-warmup-steps", type=int,   default=200)
    p.add_argument("--weight-decay",        type=float, default=0.01)
    p.add_argument("--max-grad-norm",       type=float, default=1.0)
    p.add_argument("--label-smoothing",     type=float, default=0.1)

    # checkpoint
    p.add_argument("--save-dir",            default=_DEFAULT_SAVE_DIR)
    p.add_argument("--save-every-n-steps",  type=int, default=2000)
    p.add_argument("--log-every-n-steps",   type=int, default=100)
    p.add_argument("--resume",              default=None, help="path to checkpoint dir to resume from")

    # hardware
    p.add_argument("--num-workers",        type=int,            default=4)
    p.add_argument("--prefetch-factor",    type=int,            default=2)
    p.add_argument("--persistent-workers", action="store_true", default=False)
    p.add_argument("--bf16",               action="store_true", default=True)
    p.add_argument("--no-bf16",            action="store_false", dest="bf16")
    p.add_argument("--compile",            action="store_true", default=False)
    p.add_argument("--flash-attn",         action="store_true", default=False)
    p.add_argument("--num-beams",          type=int,            default=1,
                   help="beam width for validation decoding (1 = greedy)")

    args = p.parse_args()
    cfg = argparse.Namespace(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    return cfg
