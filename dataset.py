import glob
import io
import random
import re
import struct
from typing import Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pyarrow.parquet as pq
from PIL import Image
from transformers import NougatTokenizerFast

from normalize import normalize

def _fast_image_size(data: bytes) -> tuple[int, int]:
    """Parse image dimensions from PNG/JPEG header without full decode."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        w, h = struct.unpack('>II', data[16:24])
        return w, h
    if data[:2] == b'\xff\xd8':
        i = 2
        while i < len(data) - 8:
            if data[i] != 0xff:
                break
            marker = data[i + 1]
            if marker in (0xc0, 0xc1, 0xc2):
                h, w = struct.unpack('>HH', data[i + 5:i + 9])
                return w, h
            seg_len = struct.unpack('>H', data[i + 2:i + 4])[0]
            i += 2 + seg_len
    # fallback to PIL for other formats (BMP, TIFF, etc.)
    with Image.open(io.BytesIO(data)) as img:
        return img.size


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
    while (new_w // patch_size) * (new_h // patch_size) > max_patches:
        if new_w >= new_h:
            new_w -= patch_size
        else:
            new_h -= patch_size
        new_w = max(patch_size, new_w)
        new_h = max(patch_size, new_h)
    arr = np.array(img)
    arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(arr)


class Nav2TexDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config    = config
        self.transform = transform
        self.tokenizer = NougatTokenizerFast.from_pretrained(config.tokenizer_dir)
        self.max_latex_chars = getattr(config, "max_latex_chars", 1024)
        self.max_patches     = getattr(config, "max_patches", 576)
        self.patch_size      = getattr(config, "patch_size", 16)

        globs  = config.data_glob if isinstance(config.data_glob, list) else [config.data_glob]
        ratios = getattr(config, "data_glob_ratios", None) or [1.0] * len(globs)
        if len(ratios) != len(globs):
            raise ValueError(f"data_glob_ratios length {len(ratios)} != data_glob length {len(globs)}")

        rows = []
        for pattern, ratio in zip(globs, ratios):
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No parquet files found: {pattern}")
            rng = random.Random(42)
            for f in files:
                table = pq.read_table(f, columns=["image", "latex"])
                img_list   = table["image"].to_pylist()
                latex_list = table["latex"].to_pylist()
                pairs = [
                    (img, lat) for img, lat in zip(img_list, latex_list)
                    if lat and isinstance(lat, str) and lat.strip()
                    and len(lat) <= self.max_latex_chars
                ]
                if ratio < 1.0:
                    rng.shuffle(pairs)
                    pairs = pairs[:max(1, int(len(pairs) * ratio))]
                rows.extend(pairs)

        self.samples = rows
        self._scores = [_complexity_score(latex) for _, latex in rows]

        # pre-compute approximate patch counts for bucket sampler
        # parse PNG/JPEG header bytes directly — avoids full PIL decode (~10x faster)
        ps = self.patch_size
        mp = self.max_patches
        patch_counts = []
        for img_bytes, _ in rows:
            w, h = _fast_image_size(img_bytes)
            scale = (mp * ps * ps / max(w * h, 1)) ** 0.5
            nw = max(ps, round(w * scale / ps) * ps)
            nh = max(ps, round(h * scale / ps) * ps)
            while (nw // ps) * (nh // ps) > mp:
                if nw >= nh: nw -= ps
                else:        nh -= ps
                nw = max(ps, nw); nh = max(ps, nh)
            patch_counts.append((nw // ps) * (nh // ps))
        self._patch_counts = patch_counts

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

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + getattr(self, "_epoch", 0))

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


class BucketBatchSampler(Sampler):
    """Group samples by patch count into buckets to minimise padding waste."""

    def __init__(self, dataset: Nav2TexDataset, batch_size: int,
                 patch_size: int = 16, n_buckets: int = 16, seed: int = 42):
        self.batch_size = batch_size
        self.seed       = seed

        # use pre-computed patch counts from dataset
        patch_counts = dataset._patch_counts
        sorted_idx = sorted(range(len(patch_counts)), key=lambda i: patch_counts[i])

        # split into n_buckets buckets, build batch list
        bucket_size = max(batch_size, len(sorted_idx) // n_buckets)
        self._batches: list[list[int]] = []
        for start in range(0, len(sorted_idx), bucket_size):
            bucket = sorted_idx[start : start + bucket_size]
            for b in range(0, len(bucket), batch_size):
                batch = bucket[b : b + batch_size]
                if len(batch) == batch_size:
                    self._batches.append(batch)

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + getattr(self, "_epoch", 0))
        order = list(range(len(self._batches)))
        rng.shuffle(order)
        for i in order:
            batch = self._batches[i].copy()
            rng.shuffle(batch)
            yield batch


def collate_fn(batch: list[dict], pad_token_id: int = 1, patch_size: int = 16) -> dict:
    max_text_len = max(item["input_ids"].size(0) for item in batch)

    imgs = [item["pixel_values"] for item in batch]
    if isinstance(imgs[0], torch.Tensor):
        def _ceil_patch(x): return ((x + patch_size - 1) // patch_size) * patch_size
        heights = torch.tensor([t.shape[1] for t in imgs])
        widths  = torch.tensor([t.shape[2] for t in imgs])
        max_h   = _ceil_patch(heights.max().item())
        max_w   = _ceil_patch(widths.max().item())

        padded_imgs = torch.stack([
            torch.nn.functional.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]))
            for t in imgs
        ])

        patch_h  = max_h // patch_size
        patch_w  = max_w // patch_size
        valid_ph = heights // patch_size
        valid_pw = widths   // patch_size
        row_mask = torch.arange(patch_h).unsqueeze(0) < valid_ph.unsqueeze(1)  # (B, patch_h)
        col_mask = torch.arange(patch_w).unsqueeze(0) < valid_pw.unsqueeze(1)  # (B, patch_w)
        encoder_key_masks = (row_mask.unsqueeze(2) & col_mask.unsqueeze(1)).reshape(len(imgs), -1)

        pixel_values = padded_imgs
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
    dataset = Nav2TexDataset(config, transform=transform)
    pw = getattr(config, "persistent_workers", False) and config.num_workers > 0
    pf = getattr(config, "prefetch_factor", 2) if config.num_workers > 0 else None

    _collate = _CollateFn(
        pad_token_id=config.pad_token_id,
        patch_size=getattr(config, "patch_size", 16),
    )

    patch_size = getattr(config, "patch_size", 16)

    if split == "train":
        if getattr(config, "cpe_ratio", 0) > 0:
            sampler = CPEInterleaveSampler(
                dataset,
                batch_size=config.batch_size,
                cpe_ratio=config.cpe_ratio,
            )
        else:
            sampler = BucketBatchSampler(
                dataset,
                batch_size=config.batch_size,
                patch_size=patch_size,
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
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate,
        persistent_workers=pw,
        prefetch_factor=pf,
    )
