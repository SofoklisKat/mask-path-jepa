#!/usr/bin/env python3
"""Precompute frozen DINOv2 features for the full training set (one-time, GPU).

Example:
  PYTHONPATH=. python scripts/extract_teacher_features.py \\
    --dataset cifar100 --batch-size 32 --amp \\
    --output teacher_features/cifar100_dinov2_vits14_cls.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from mask_path_jepa.teacher import IMAGENET_MEAN, IMAGENET_STD, encode_teacher_batch, load_dinov2


def resolve_global_index(base: Dataset, i: int) -> int:
    if isinstance(base, Subset):
        return resolve_global_index(base.dataset, base.indices[i])
    return i


def unwrap_dataset(base: Dataset) -> Dataset:
    while isinstance(base, Subset):
        base = base.dataset
    return base


class TeacherExtractDataset(Dataset):
    """Raw train sample -> DINOv2 input + global dataset index."""

    def __init__(self, base: Dataset, transform: transforms.Compose) -> None:
        self.base = base
        self.transform = transform
        self.root = unwrap_dataset(base)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        global_idx = resolve_global_index(self.base, i)
        if hasattr(self.root, "data"):
            img = Image.fromarray(self.root.data[global_idx])
        else:
            img, _ = self.root[global_idx]
        return self.transform(img), global_idx


def build_teacher_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute DINOv2 teacher features")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--teacher", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--pool", default="cls", choices=["cls", "mean_patch"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="fp16 forward (saves VRAM)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    if args.dataset == "cifar10":
        raw = datasets.CIFAR10(args.data_dir, train=True, download=args.download)
    else:
        raw = datasets.CIFAR100(args.data_dir, train=True, download=args.download)

    if args.train_subset is not None:
        raw = Subset(raw, range(min(args.train_subset, len(raw))))

    root = unwrap_dataset(raw)
    root_len = len(root)
    dataset = TeacherExtractDataset(raw, build_teacher_transform(args.image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    n = len(dataset)
    print(f"Loading {args.teacher} on {device} for {n} images (store size {root_len}) ...")
    model = load_dinov2(args.teacher, device)

    probe, _ = next(iter(loader))
    with torch.autocast(device.type, enabled=args.amp and device.type == "cuda"):
        feat = encode_teacher_batch(model, probe.to(device), pool=args.pool)
    dim = feat.size(1)
    features = torch.zeros(root_len, dim, dtype=torch.float32)
    filled = torch.zeros(root_len, dtype=torch.bool)

    t0 = time.time()
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=args.amp and device.type == "cuda"):
            z = encode_teacher_batch(model, images, pool=args.pool).float().cpu()
        for row, idx in enumerate(indices.tolist()):
            features[idx] = z[row]
            filled[idx] = True
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)} | stored {filled.sum().item()}/{root_len}")

    if args.train_subset is None and not filled.all():
        missing = (~filled).nonzero(as_tuple=False).view(-1)[:5].tolist()
        raise RuntimeError(f"Missing { (~filled).sum().item() } indices, e.g. {missing}")

    out = Path(args.output or f"teacher_features/{args.dataset}_{args.teacher}_{args.pool}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": features,
        "dataset": args.dataset,
        "teacher": args.teacher,
        "pool": args.pool,
        "image_size": args.image_size,
        "dim": dim,
        "num_stored": int(filled.sum().item()),
        "train_subset": args.train_subset,
    }
    torch.save(payload, out)
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps({k: v for k, v in payload.items() if k != "features"}, indent=2))
    mb = out.stat().st_size / (1024 * 1024)
    print(f"Saved {out} ({mb:.1f} MB, {n} x {dim}) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
