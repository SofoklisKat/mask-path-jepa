#!/usr/bin/env python3
"""PCA of encoder embeddings on two classes — random or trained weights.

Checks whether embeddings (even at init) show any class structure.

Example (random ResNet, CIFAR-100 classes 3 vs 5):
  PYTHONPATH=. python scripts/visualize_class_pca_thumbnails.py \\
    --dataset cifar100 --class-a 3 --class-b 5 --output results/analysis/random_pca.png

With a trained checkpoint:
  PYTHONPATH=. python scripts/visualize_class_pca_thumbnails.py \\
    --checkpoint results/paper_cifar100/latent_infonce_jepa_augment/checkpoint_best.pt \\
    --class-a 3 --class-b 5
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
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from tripletjepa.data import DATASETS
from tripletjepa.models import build_encoder


def denormalize(tensor: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    """CHW tensor -> HWC float image in [0, 1]."""
    out = tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        out[c] = out[c] * s + m
    out = out.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return out


class ClassSubsetDataset(Dataset):
    """Wrap base dataset; return normalized tensor, display tensor, label."""

    def __init__(
        self,
        base: Dataset,
        indices: list[int],
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> None:
        self.base = base
        self.indices = indices
        self.mean = mean
        self.std = std
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx = self.indices[i]
        if hasattr(self.base, "data") and hasattr(self.base, "targets"):
            # CIFAR numpy layout
            img = self.base.data[idx]
            label = int(self.base.targets[idx])
            pil = Image.fromarray(img)
        else:
            pil, label = self.base[idx]
            if isinstance(pil, torch.Tensor):
                # Already tensor — rare path
                display = pil.clamp(0, 1)
                norm = transforms.Normalize(self.mean, self.std)(display.clone())
                return norm, display, int(label)
        display = self.to_tensor(pil)
        norm = transforms.Normalize(self.mean, self.std)(display.clone())
        return norm, display, label


def collect_class_indices(targets: list[int] | np.ndarray, class_id: int, n: int) -> list[int]:
    idx = [i for i, t in enumerate(targets) if int(t) == class_id]
    if len(idx) < n:
        raise ValueError(f"class {class_id}: need {n} images, found {len(idx)}")
    return idx[:n]


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return 2D coords and fraction variance explained by 2 PCs."""
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    var = (s**2) / max((s**2).sum(), 1e-12)
    explained = float(var[:2].sum())
    return coords, explained


def pairwise_stats(z: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    z = F.normalize(z, dim=-1)
    sim = z @ z.T
    n = z.size(0)
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            v = sim[i, j].item()
            if labels[i] == labels[j]:
                within.append(v)
            else:
                between.append(v)
    return {
        "mean_cosine_within": float(np.mean(within)) if within else float("nan"),
        "mean_cosine_between": float(np.mean(between)) if between else float("nan"),
        "gap": float(np.mean(within) - np.mean(between)) if within and between else float("nan"),
    }


def add_thumbnails(
    ax: plt.Axes,
    coords: np.ndarray,
    images: list[np.ndarray],
    labels: np.ndarray,
    class_a: int,
    zoom: float,
) -> None:
    colors = {class_a: "#2980b9", int(labels.max() if labels.min() == class_a else labels[labels != class_a][0]): "#c0392b"}
    for (x, y), img, lab in zip(coords, images, labels):
        im = OffsetImage(img, zoom=zoom)
        imbox = AnnotationBbox(
            im,
            (x, y),
            frameon=True,
            pad=0.02,
            bboxprops=dict(edgecolor=colors.get(int(lab), "gray"), linewidth=1.5),
        )
        ax.add_artist(imbox)


def build_random_encoder(cfg: dict | None, spec, args: argparse.Namespace, device: torch.device):
    embed_dim = args.embed_dim if cfg is None else cfg.get("embed_dim", args.embed_dim)
    backbone = args.backbone if cfg is None else cfg.get("backbone", args.backbone)
    torch.manual_seed(args.seed)
    encoder = build_encoder(
        backbone,
        in_channels=spec.in_channels,
        embed_dim=embed_dim,
        image_size=spec.image_size,
        vit_patch_size=args.vit_patch_size if cfg is None else cfg.get("vit_patch_size", 4),
        vit_depth=args.vit_depth if cfg is None else cfg.get("vit_depth", 6),
        vit_heads=args.vit_heads if cfg is None else cfg.get("vit_heads", 4),
        vit_mlp_dim=args.vit_mlp_dim if cfg is None else cfg.get("vit_mlp_dim", 512),
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
        state = ckpt["model"]
        enc_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
        encoder.load_state_dict(enc_state)
        title_suffix = f"trained ({Path(args.checkpoint).parent.name})"
    else:
        title_suffix = "random init"
    encoder.eval()
    return encoder, title_suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="2-class PCA with image thumbnails")
    parser.add_argument("--dataset", default="cifar100", choices=sorted(DATASETS))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--class-a", type=int, default=3)
    parser.add_argument("--class-b", type=int, default=5)
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--checkpoint", default=None, help="Optional; default = random weights")
    parser.add_argument("--backbone", default="resnet", choices=["resnet", "vit"])
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--vit-patch-size", type=int, default=4)
    parser.add_argument("--vit-depth", type=int, default=6)
    parser.add_argument("--vit-heads", type=int, default=4)
    parser.add_argument("--vit-mlp-dim", type=int, default=512)
    parser.add_argument("--output", default="results/analysis/class_pca_thumbnails.png")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--thumbnail-zoom", type=float, default=0.45)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    if args.device != "auto":
        device = torch.device(args.device)

    spec = DATASETS[args.dataset]
    cfg = None
    if args.checkpoint:
        cfg = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)["config"]

    # Raw CIFAR for index lookup (no aug)
    if args.dataset == "cifar10":
        raw = datasets.CIFAR10(args.data_dir, train=True, download=args.download)
    elif args.dataset == "cifar100":
        raw = datasets.CIFAR100(args.data_dir, train=True, download=args.download)
    else:
        raise ValueError("Currently supports cifar10/cifar100 for thumbnail source.")

    targets = raw.targets
    idx_a = collect_class_indices(targets, args.class_a, args.n_per_class)
    idx_b = collect_class_indices(targets, args.class_b, args.n_per_class)
    all_indices = idx_a + idx_b

    subset = ClassSubsetDataset(raw, all_indices, spec.mean, spec.std)
    norms, displays, labels = [], [], []
    for i in range(len(subset)):
        n, d, lab = subset[i]
        norms.append(n)
        displays.append(d)
        labels.append(lab)
    x_norm = torch.stack(norms)
    labels_t = torch.tensor(labels)
    images_hwc = [denormalize(x_norm[i], spec.mean, spec.std) for i in range(len(labels))]

    encoder, title_suffix = build_random_encoder(cfg, spec, args, device)
    with torch.no_grad():
        z = encoder(x_norm.to(device)).cpu()

    z_np = z.numpy()
    pixels = x_norm.view(x_norm.size(0), -1).numpy()

    coords_enc, expl_enc = pca_2d(z_np)
    coords_pix, expl_pix = pca_2d(pixels)

    stats_enc = pairwise_stats(z, labels_t)
    stats_pix = pairwise_stats(F.normalize(torch.tensor(pixels), dim=-1), labels_t)

    labels_np = labels_t.numpy()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, coords, expl, name, stats in (
        (axes[0], coords_enc, expl_enc, f"Encoder ({title_suffix})", stats_enc),
        (axes[1], coords_pix, expl_pix, "Raw pixels (baseline)", stats_pix),
    ):
        ax.scatter(
            coords[labels_np == args.class_a, 0],
            coords[labels_np == args.class_a, 1],
            s=30,
            c="#2980b9",
            alpha=0.4,
            label=f"class {args.class_a}",
        )
        ax.scatter(
            coords[labels_np == args.class_b, 0],
            coords[labels_np == args.class_b, 1],
            s=30,
            c="#c0392b",
            alpha=0.4,
            label=f"class {args.class_b}",
        )
        add_thumbnails(ax, coords, images_hwc, labels_np, args.class_a, args.thumbnail_zoom)
        ax.set_title(
            f"{name}\nPCA var explained: {expl:.1%} | "
            f"cos gap: {stats['gap']:.3f} (within {stats['mean_cosine_within']:.3f}, "
            f"between {stats['mean_cosine_between']:.3f})"
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"{args.dataset}: {args.n_per_class} images x class {args.class_a} vs {args.class_b}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    summary = {
        "dataset": args.dataset,
        "class_a": args.class_a,
        "class_b": args.class_b,
        "n_per_class": args.n_per_class,
        "weights": title_suffix,
        "encoder_stats": stats_enc,
        "pixel_stats": stats_pix,
        "pca_var_explained_encoder": expl_enc,
        "pca_var_explained_pixels": expl_pix,
        "interpretation": (
            "Visible class separation in PCA + positive cosine gap suggests signal to exploit. "
            "Random encoder often shows weak pixel baseline; trained encoder should increase gap."
        ),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
