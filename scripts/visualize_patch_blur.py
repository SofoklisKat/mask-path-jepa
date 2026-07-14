#!/usr/bin/env python3
"""Visualize patch-blur curriculum: initial vs final corrupt views."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from tripletjepa.views import make_corrupt_view


def draw_patch_grid(ax, patch_size: int, color: str = "yellow", alpha: float = 0.35) -> None:
    """Overlay patch grid to show corruption is tile-local, not whole-image."""
    for x in range(0, 33, patch_size):
        ax.axvline(x - 0.5, color=color, linewidth=0.4, alpha=alpha)
    for y in range(0, 33, patch_size):
        ax.axhline(y - 0.5, color=color, linewidth=0.4, alpha=alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=42, help="CIFAR train image index")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/patch_blur_curriculum"),
    )
    parser.add_argument("--ratio-min", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.6, help="Final patch-blur ratio")
    parser.add_argument("--sigma-min", type=float, default=0.5)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--patch-size", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
    img, _ = ds[args.index]
    x = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)

    common = dict(
        schedule="mask_curriculum",
        patch_blur_ratio_min=args.ratio_min,
        mask_ratio=args.mask_ratio,
        blur_sigma_min=args.sigma_min,
        blur_sigma_max=args.sigma_max,
        min_block=args.patch_size,
    )
    initial = make_corrupt_view(x, progress=0.0, **common)
    final = make_corrupt_view(x, progress=1.0, **common)

    def to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    clean, init_np, final_np = to_numpy(x), to_numpy(initial), to_numpy(final)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5))
    axes[0, 0].imshow(clean)
    axes[0, 0].set_title("Clean view")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(clean)
    draw_patch_grid(axes[0, 1], args.patch_size)
    axes[0, 1].set_title(f"Patch grid ({args.patch_size}×{args.patch_size} tiles)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(init_np)
    draw_patch_grid(axes[1, 0], args.patch_size)
    axes[1, 0].set_title(
        f"(a) Initial — epoch 1\n{args.ratio_min:.0%} tiles blurred, σ={args.sigma_min}"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(final_np)
    draw_patch_grid(axes[1, 1], args.patch_size)
    axes[1, 1].set_title(
        f"(b) Final — epoch 100\n{args.mask_ratio:.0%} tiles blurred, σ={args.sigma_max}"
    )
    axes[1, 1].axis("off")

    fig.suptitle(
        "Patch-local blur curriculum (selected tiles only, not whole image)",
        fontsize=11,
    )
    fig.tight_layout()

    out = args.output_dir / "patch_blur_curriculum_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
