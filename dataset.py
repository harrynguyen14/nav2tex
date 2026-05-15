import glob
import io
import json
import random
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pyarrow.parquet as pq
from PIL import Image
from transformers import NougatTokenizerFast

from normalize import normalize

_CPE_PATTERNS = re.compile(
    r'\\(frac|int|sum|prod|matrix|pmatrix|bmatrix|cases|align|begin|sqrt'
    r'|underbrace|overbrace|overset|underset|substack|bigoplus|bigotimes'
    r'|lim|sup|inf|max|min)\b'
)


def _complexity_score(s: str) -> float:
    return len(s) + len(_CPE_PATTERNS.findall(s)) * 15 + s.count('{') * 10


def _patch_aware_resize(img: Image.Image, max_patches: int, patch_size: int = 16) -> Image.Image:
    w, h = img.size
    scale = (max_patches * patch_size * patch_size / (w * h)) ** 0.5
    new_w = max(patch_size, round(w * scale / patch_size) * patch_size)
    new_h = max(patch_size, round(h * scale / patch_size) * patch_size)
    # clamp so neither dim exceeds what fits in max_patches
    while (new_w // patch_size) * (new_h // patch_size) > max_patches:
        if new_w >= new_h:
            new_w -= patch_size
        else:
            new_h -= patch_size
        new_w = max(patch_size, new_w)
        new_h = max(patch_size, new_h)
    return img.resize((new_w, new_h), Image.BICUBIC)


class Nav2TexDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config    = config
        self.transform = transform
        self.tokenizer = NougatTokenizerFast.from_pretrained(config.tokenizer_dir)
        self.max_latex_chars = getattr(config, "max_latex_chars", 1024)
        self.max_patches     = getattr(config, "max_patches", 576)
        self.patch_size      = getattr(config, "patch_size", 16)

        globs = config.data_glob if isinstance(config.data_glob, list) else [config.data_glob]
        files = sorted(f for pattern in globs for f in glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No parquet files found: {config.data_glob}")

        rows = []
        for f in files:
            table = pq.read_table(f, columns=["image", "latex"])
            for img_bytes, latex in zip(table["image"].to_pylist(), table["latex"].to_pylist()):
                if not latex or not isinstance(latex, str) or not latex.strip():
                    continue
                if len(latex) > self.max_latex_chars:
                    continue
                rows.append((img_bytes, latex))

        self.samples = rows
        self._scores = [_complexity_score(latex) for _, latex in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_bytes, latex = self.samples[idx]

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = _patch_aware_resize(img, self.max_patches, self.patch_size)

        if self.transform is not None:
            pixel_values = self.transform(img)
        else:
            pixel_values = img

        text = normalize(latex)
        ids  = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)

        max_tokens = self.config.max_seq_len - 1
        ids = ids[:max_tokens]

        input_ids = [self.config.bos_token_id] + ids
        labels    = ids + [self.config.eos_token_id]

        return {
            "pixel_values": pixel_values,
            "input_ids":    torch.tensor(input_ids, dtype=torch.long),
            "labels":       torch.tensor(labels,    dtype=torch.long),
            "true_len":     torch.tensor(len(ids) / self.config.max_seq_len, dtype=torch.float),
        }

    def normal_indices(self) -> list[int]:
        t = self.config.cpe_score_threshold
        return [i for i, sc in enumerate(self._scores) if sc <= t]

    def cpe_indices(self) -> list[int]:
        t = self.config.cpe_score_threshold
        return [i for i, sc in enumerate(self._scores) if sc > t]

    def score_stats(self) -> dict:
        import statistics
        scores = self._scores
        thresh = self.config.cpe_score_threshold
        n_cpe  = sum(1 for sc in scores if sc > thresh)
        return {
            "total":   len(scores),
            "n_cpe":   n_cpe,
            "n_spe":   len(scores) - n_cpe,
            "cpe_pct": round(n_cpe / len(scores) * 100, 2),
            "median":  round(statistics.median(scores), 1),
            "p95":     round(sorted(scores)[int(len(scores) * 0.95)], 1),
        }


class Nav2TexDecodedDataset(Dataset):
    """Fast dataset reading pre-decoded shards from predecode.py."""

    def __init__(self, config, transform=None):
        self.config    = config
        self.transform = transform

        globs = config.data_glob if isinstance(config.data_glob, list) else [config.data_glob]
        shard_dirs = sorted(p for pattern in globs for p in glob.glob(pattern))
        if not shard_dirs:
            raise FileNotFoundError(f"No decoded shards found: {config.data_glob}")

        self._img_paths: list[Path] = []
        self._input_ids: list[np.ndarray] = []
        self._labels:    list[np.ndarray] = []
        self._true_lens: list[float]      = []
        self._scores:    list[float]      = []

        for shard_dir in shard_dirs:
            shard_dir = Path(shard_dir)
            meta_file = shard_dir / "meta.json"
            if not meta_file.exists():
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            n = meta["num_samples"]
            tokens = np.load(shard_dir / "tokens.npz")
            input_ids = tokens["input_ids"]   # (N, max_seq_len) int32
            labels    = tokens["labels"]       # (N, max_seq_len) int32
            lengths   = tokens["lengths"]      # (N,) int32
            true_len  = tokens["true_len"]     # (N,) float32
            scores    = tokens["scores"]       # (N,) float32
            for i in range(n):
                seq_len = int(lengths[i])
                self._img_paths.append(shard_dir / "images" / f"{i:07d}.npy")
                self._input_ids.append(input_ids[i, :seq_len])
                self._labels.append(labels[i, :seq_len])
                self._true_lens.append(float(true_len[i]))
                self._scores.append(float(scores[i]))

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self, idx: int) -> dict:
        img_np    = np.load(self._img_paths[idx])   # (H, W, 3) uint8
        img       = Image.fromarray(img_np)
        if self.transform is not None:
            pixel_values = self.transform(img)
        else:
            from torchvision.transforms.functional import to_tensor
            pixel_values = to_tensor(img)

        input_ids = self._input_ids[idx].astype(np.int64)
        labels    = self._labels[idx].astype(np.int64)

        return {
            "pixel_values": pixel_values,
            "input_ids":    torch.from_numpy(input_ids),
            "labels":       torch.from_numpy(labels),
            "true_len":     torch.tensor(self._true_lens[idx], dtype=torch.float),
        }

    def normal_indices(self) -> list[int]:
        t = self.config.cpe_score_threshold
        return [i for i, sc in enumerate(self._scores) if sc <= t]

    def cpe_indices(self) -> list[int]:
        t = self.config.cpe_score_threshold
        return [i for i, sc in enumerate(self._scores) if sc > t]

    def score_stats(self) -> dict:
        import statistics
        scores = self._scores
        thresh = self.config.cpe_score_threshold
        n_cpe  = sum(1 for sc in scores if sc > thresh)
        return {
            "total":   len(scores),
            "n_cpe":   n_cpe,
            "n_spe":   len(scores) - n_cpe,
            "cpe_pct": round(n_cpe / len(scores) * 100, 2),
            "median":  round(statistics.median(scores), 1),
            "p95":     round(sorted(scores)[int(len(scores) * 0.95)], 1),
        }


class CPEInterleaveSampler(Sampler):
    def __init__(self, dataset: Nav2TexDataset, batch_size: int, cpe_ratio: float, seed: int = 42):
        self.normal_idx = dataset.normal_indices()
        self.cpe_idx    = dataset.cpe_indices()
        self.batch_size = batch_size
        self.cpe_ratio  = cpe_ratio
        self.seed       = seed

        self.n_cpe_per_batch    = max(1, round(batch_size * cpe_ratio))
        self.n_normal_per_batch = batch_size - self.n_cpe_per_batch
        self.n_batches          = len(self.normal_idx) // self.n_normal_per_batch

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)

        normal_pool = self.normal_idx.copy()
        rng.shuffle(normal_pool)

        cpe_pool = self.cpe_idx.copy()
        rng.shuffle(cpe_pool)
        if len(cpe_pool) < self.n_batches * self.n_cpe_per_batch:
            repeats  = (self.n_batches * self.n_cpe_per_batch) // max(len(cpe_pool), 1) + 1
            cpe_pool = (cpe_pool * repeats)[: self.n_batches * self.n_cpe_per_batch]
            rng.shuffle(cpe_pool)

        for b in range(self.n_batches):
            n_start   = b * self.n_normal_per_batch
            c_start   = b * self.n_cpe_per_batch
            batch_idx = (
                normal_pool[n_start : n_start + self.n_normal_per_batch]
                + cpe_pool[c_start : c_start + self.n_cpe_per_batch]
            )
            rng.shuffle(batch_idx)
            yield batch_idx


def collate_fn(batch: list[dict], pad_token_id: int = 1, patch_size: int = 16) -> dict:
    max_text_len = max(item["input_ids"].size(0) for item in batch)

    # image padding: find max H and W in batch, pad to multiple of patch_size
    imgs = [item["pixel_values"] for item in batch]
    if isinstance(imgs[0], torch.Tensor):
        def _ceil_patch(x): return ((x + patch_size - 1) // patch_size) * patch_size
        max_h = _ceil_patch(max(t.shape[1] for t in imgs))
        max_w = _ceil_patch(max(t.shape[2] for t in imgs))
        padded_imgs = []
        encoder_key_masks = []
        for t in imgs:
            _, h, w = t.shape
            pad_h = max_h - h
            pad_w = max_w - w
            padded = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=0.0)
            padded_imgs.append(padded)
            # True = valid patch, False = padding patch
            patch_h = max_h // patch_size
            patch_w = max_w // patch_size
            valid_ph = h // patch_size
            valid_pw = w // patch_size
            mask = torch.zeros(patch_h, patch_w, dtype=torch.bool)
            mask[:valid_ph, :valid_pw] = True
            encoder_key_masks.append(mask.flatten())
        pixel_values      = torch.stack(padded_imgs)
        encoder_key_masks = torch.stack(encoder_key_masks)
    else:
        pixel_values      = None
        encoder_key_masks = None

    input_ids_list, labels_list, attn_mask_list = [], [], []
    for item in batch:
        n   = item["input_ids"].size(0)
        pad = max_text_len - n
        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
        attn_mask_list.append(
            torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
        )

    return {
        "pixel_values":     pixel_values,
        "encoder_key_mask": encoder_key_masks,
        "input_ids":        torch.stack(input_ids_list),
        "labels":           torch.stack(labels_list),
        "attention_mask":   torch.stack(attn_mask_list),
        "true_len":         torch.stack([item["true_len"] for item in batch]),
    }


class _CollateFn:
    def __init__(self, pad_token_id: int, patch_size: int):
        self.pad_token_id = pad_token_id
        self.patch_size   = patch_size

    def __call__(self, batch):
        return collate_fn(batch, pad_token_id=self.pad_token_id, patch_size=self.patch_size)


def build_dataloader(config, transform=None, split: str = "train") -> DataLoader:
    if getattr(config, "decoded", False):
        dataset = Nav2TexDecodedDataset(config, transform=transform)
    else:
        dataset = Nav2TexDataset(config, transform=transform)
    pw = getattr(config, "persistent_workers", False) and config.num_workers > 0
    pf = getattr(config, "prefetch_factor", 2) if config.num_workers > 0 else None

    _collate = _CollateFn(
        pad_token_id=config.pad_token_id,
        patch_size=getattr(config, "patch_size", 16),
    )

    if split == "train" and getattr(config, "cpe_ratio", 0) > 0:
        sampler = CPEInterleaveSampler(
            dataset,
            batch_size=config.batch_size,
            cpe_ratio=config.cpe_ratio,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=_collate,
            persistent_workers=pw,
            prefetch_factor=pf,
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
        persistent_workers=pw,
        prefetch_factor=pf,
    )