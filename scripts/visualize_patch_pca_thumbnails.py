#!/usr/bin/env python3
"""PCA of patch-level encoder embeddings (random or trained weights).

Splits each image into a grid of patches, encodes every patch, and plots PCA
with patch thumbnails for two classes.

Example:
  PYTHONPATH=. python scripts/visualize_patch_pca_thumbnails.py \\
    --dataset cifar100 --class-a 3 --class-b 5 \\
    --output results/analysis/random_patch_pca.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from tripletjepa.data import DATASETS
from tripletjepa.models import build_encoder


def extract_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """CHW image -> (num_patches, C, patch_size, patch_size) row-major grid."""
    c, h, w = image.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image ({h},{w}) not divisible by patch_size={patch_size}")
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    return patches.permute(1, 2, 0, 3, 4).reshape(-1, c, patch_size, patch_size)


def prepare_encoder_input(patches: torch.Tensor, upsample_to: int | None) -> torch.Tensor:
    if upsample_to is None:
        return patches
    return F.interpolate(patches, size=(upsample_to, upsample_to), mode="bilinear", align_corners=False)


class IndexedCifar(Dataset):
    def __init__(self, base, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.base = base
        self.mean = mean
        self.std = std
        self.to_tensor = transforms.ToTensor()

    def get_raw(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img = self.base.data[index]
        label = int(self.base.targets[index])
        display = self.to_tensor(Image.fromarray(img))
        norm = transforms.Normalize(self.mean, self.std)(display.clone())
        return norm, display, label


def collect_indices(targets: list[int], class_id: int, n: int) -> list[int]:
    idx = [i for i, t in enumerate(targets) if int(t) == class_id]
    if len(idx) < n:
        raise ValueError(f"class {class_id}: need {n}, found {len(idx)}")
    return idx[:n]


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    var = (s**2) / max((s**2).sum(), 1e-12)
    return coords, float(var[:2].sum())


def cosine_gap(z: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    z = F.normalize(z, dim=-1)
    sim = z @ z.T
    within, between = [], []
    for i in range(z.size(0)):
        for j in range(i + 1, z.size(0)):
            v = sim[i, j].item()
            (within if labels[i] == labels[j] else between).append(v)
    return {
        "mean_cosine_within": float(np.mean(within)) if within else float("nan"),
        "mean_cosine_between": float(np.mean(between)) if between else float("nan"),
        "gap": float(np.mean(within) - np.mean(between)) if within and between else float("nan"),
    }


def add_patch_thumbnails(ax, coords, patch_imgs, zoom, edgecolor) -> None:
    for (x, y), pimg in zip(coords, patch_imgs):
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(pimg, zoom=zoom),
                (x, y),
                frameon=True,
                pad=0.01,
                bboxprops=dict(edgecolor=edgecolor, linewidth=1.0),
            )
        )


def load_encoder(args, spec, device):
    cfg = None
    if args.checkpoint:
        cfg = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)["config"]
    torch.manual_seed(args.seed)
    encoder = build_encoder(
        args.backbone if cfg is None else cfg.get("backbone", args.backbone),
        in_channels=spec.in_channels,
        embed_dim=args.embed_dim if cfg is None else cfg.get("embed_dim", args.embed_dim),
        image_size=spec.image_size,
        vit_patch_size=getattr(args, "vit_patch_size", 4),
        vit_depth=getattr(args, "vit_depth", 6),
        vit_heads=getattr(args, "vit_heads", 4),
        vit_mlp_dim=getattr(args, "vit_mlp_dim", 512),
    ).to(device)
    suffix = "random init"
    if args.checkpoint:
        state = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)["model"]
        enc_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
        encoder.load_state_dict(enc_state)
        suffix = f"trained ({Path(args.checkpoint).parent.name})"
    encoder.eval()
    return encoder, suffix


@torch.no_grad()
def encode_patches(encoder, images_norm, patch_size, upsample_to, device):
    all_z, all_thumbs = [], []
    for img in images_norm:
        patches = extract_patches(img, patch_size)
        z = encoder(prepare_encoder_input(patches, upsample_to).to(device)).cpu()
        all_z.append(z)
        all_thumbs.append([p.permute(1, 2, 0).clamp(0, 1).numpy() for p in patches])
    return torch.cat(all_z, dim=0), all_thumbs


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch-level PCA with thumbnails")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--class-a", type=int, default=3)
    parser.add_argument("--class-b", type=int, default=5)
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--upsample-to", type=int, default=32, help="0 = native patch size")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backbone", default="resnet", choices=["resnet", "vit"])
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--output", default="results/analysis/patch_pca_thumbnails.png")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patch-thumb-zoom", type=float, default=0.55)
    args = parser.parse_args()

    upsample = None if args.upsample_to <= 0 else args.upsample_to
    device = torch.device(args.device)
    spec = DATASETS[args.dataset]
    raw = (
        datasets.CIFAR100(args.data_dir, train=True, download=args.download)
        if args.dataset == "cifar100"
        else datasets.CIFAR10(args.data_dir, train=True, download=args.download)
    )
    ds = IndexedCifar(raw, spec.mean, spec.std)
    idx_a = collect_indices(raw.targets, args.class_a, args.n_per_class)
    idx_b = collect_indices(raw.targets, args.class_b, args.n_per_class)
    indices = idx_a + idx_b
    labels = torch.tensor([args.class_a] * len(idx_a) + [args.class_b] * len(idx_b))

    norms, displays = [], []
    for idx in indices:
        n, d, _ = ds.get_raw(idx)
        norms.append(n)
        displays.append(d)
    images_norm = torch.stack(norms)

    encoder, suffix = load_encoder(args, spec, device)
    ppg = (spec.image_size // args.patch_size) ** 2

    with torch.no_grad():
        z_full = encoder(images_norm.to(device)).cpu()
    z_patches, patch_thumbs = encode_patches(
        encoder, images_norm, args.patch_size, upsample, device
    )
    patch_labels = labels.repeat_interleave(ppg)
    z_mean = z_patches.view(len(indices), ppg, -1).mean(dim=1)

    stats_full = cosine_gap(z_full, labels)
    stats_patches = cosine_gap(z_patches, patch_labels)
    stats_mean = cosine_gap(z_mean, labels)

    coords_all, expl_all = pca_2d(z_patches.numpy())
    ex_a, ex_b = 0, args.n_per_class

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])

    for col, (img_idx, cls, color) in enumerate(
        ((ex_a, args.class_a, "#2980b9"), (ex_b, args.class_b, "#c0392b"))
    ):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(displays[img_idx].permute(1, 2, 0).numpy())
        gh = spec.image_size // args.patch_size
        for g in range(1, gh):
            ax.axhline(g * args.patch_size - 0.5, color="yellow", lw=0.4, alpha=0.8)
            ax.axvline(g * args.patch_size - 0.5, color="yellow", lw=0.4, alpha=0.8)
        ax.set_title(f"Class {cls} — {args.patch_size}x{args.patch_size} grid")
        ax.axis("off")

    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")
    ax_leg.text(
        0.0,
        0.85,
        f"Encoder: {suffix}\npatch={args.patch_size}, upsample={upsample or 'native'}\n\n"
        f"full image gap: {stats_full['gap']:.3f}\n"
        f"all patches gap: {stats_patches['gap']:.3f}\n"
        f"mean patch/img gap: {stats_mean['gap']:.3f}",
        fontsize=11,
        va="top",
        family="monospace",
    )

    ax = fig.add_subplot(gs[1, 0])
    for cls, color in ((args.class_a, "#2980b9"), (args.class_b, "#c0392b")):
        m = patch_labels.numpy() == cls
        ax.scatter(coords_all[m, 0], coords_all[m, 1], s=6, c=color, alpha=0.35, label=f"class {cls}")
    ax.set_title(f"All {len(z_patches)} patch embeddings PCA (var={expl_all:.0%})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    for col, (img_idx, cls, color) in enumerate(
        ((ex_a, args.class_a, "#2980b9"), (ex_b, args.class_b, "#c0392b")),
        start=1,
    ):
        ax = fig.add_subplot(gs[1, col])
        start = img_idx * ppg
        pc = coords_all[start : start + ppg]
        add_patch_thumbnails(ax, pc, patch_thumbs[img_idx], args.patch_thumb_zoom, color)
        ax.scatter(pc[:, 0], pc[:, 1], s=8, c=color, alpha=0.3)
        ax.set_title(f"Class {cls}: 1 image, {ppg} patch thumbnails")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{args.dataset} patch PCA — classes {args.class_a} vs {args.class_b}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

    bar_path = out.with_name(out.stem + "_gaps.png")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(["full image", "mean patches", "all patches"], [stats_full["gap"], stats_mean["gap"], stats_patches["gap"]])
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("cosine gap (within - between class)")
    ax2.set_title(f"Class separation — {suffix}")
    fig2.tight_layout()
    fig2.savefig(bar_path, dpi=140)
    plt.close(fig2)

    summary = {
        "weights": suffix,
        "patch_size": args.patch_size,
        "upsample_to": upsample,
        "patches_per_image": ppg,
        "stats_full_image": stats_full,
        "stats_all_patches": stats_patches,
        "stats_mean_patch_per_image": stats_mean,
    }
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out} and {bar_path}")


if __name__ == "__main__":
    main()
