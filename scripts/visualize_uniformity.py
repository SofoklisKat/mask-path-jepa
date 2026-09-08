#!/usr/bin/env python3
"""Visualize hypersphere uniformity on frozen encoder embeddings.

Uniformity (Wang & Isola): spread points on the unit sphere so pairwise
distances are large. Low loss = more uniform; collapsed cloud = high loss.

Example:
  PYTHONPATH=. python scripts/visualize_uniformity.py \\
    --checkpoint results/paper_cifar100/latent_infonce_jepa_augment/checkpoint_best.pt \\
    --output-dir results/analysis/infonce/uniformity_viz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mask_path_jepa.data import get_dataloaders
from mask_path_jepa.eval import extract_features
from mask_path_jepa.losses import uniformity_loss
from mask_path_jepa.models import JEPA


def load_encoder(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    _, _, spec = get_dataloaders(
        cfg["dataset"],
        cfg.get("data_dir", "./data"),
        batch_size=256,
        num_workers=2,
        download=False,
    )
    model = JEPA(
        in_channels=spec.in_channels,
        embed_dim=cfg["embed_dim"],
        use_ema_target=cfg.get("use_ema_target", True),
        backbone=cfg.get("backbone", "resnet"),
        image_size=spec.image_size,
        vit_patch_size=cfg.get("vit_patch_size", 4),
        vit_depth=cfg.get("vit_depth", 6),
        vit_heads=cfg.get("vit_heads", 4),
        vit_mlp_dim=cfg.get("vit_mlp_dim", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.encoder, cfg, spec


def pairwise_cosine(z: torch.Tensor, max_pairs: int = 80_000, seed: int = 42) -> np.ndarray:
    z = F.normalize(z, dim=-1)
    n = z.size(0)
    gen = torch.Generator().manual_seed(seed)
    if n * (n - 1) // 2 <= max_pairs:
        sim = z @ z.T
        mask = ~torch.eye(n, dtype=torch.bool)
        return sim[mask].numpy()
    idx_a = torch.randint(0, n, (max_pairs,), generator=gen)
    idx_b = torch.randint(0, n, (max_pairs,), generator=gen)
    mask = idx_a != idx_b
    return (z[idx_a[mask]] * z[idx_b[mask]]).sum(dim=-1).numpy()


def project_pca_2d(z: torch.Tensor) -> np.ndarray:
    x = F.normalize(z, dim=-1).numpy()
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def synthetic_sphere(n: int, dim: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, dim, generator=gen)
    return F.normalize(z, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize embedding uniformity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="./results/analysis/uniformity_viz")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--uniformity-t", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    if args.device != "auto":
        device = torch.device(args.device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    encoder, cfg, spec = load_encoder(Path(args.checkpoint), device)
    train_loader, _, _ = get_dataloaders(
        cfg["dataset"],
        cfg.get("data_dir", "./data"),
        batch_size=256,
        num_workers=2,
        download=False,
    )
    x, y = extract_features(encoder, train_loader, device)
    if x.size(0) > args.max_samples:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(x.size(0), generator=g)[: args.max_samples]
        x, y = x[idx], y[idx]

    x_norm = F.normalize(x, dim=-1)
    unif_val = uniformity_loss(x_norm, t=args.uniformity_t).item()
    rand_sphere = synthetic_sphere(x.size(0), x.size(1), seed=args.seed)
    unif_random = uniformity_loss(rand_sphere, t=args.uniformity_t).item()

    cos_emb = pairwise_cosine(x, seed=args.seed)
    cos_rand = pairwise_cosine(rand_sphere, seed=args.seed)

    # --- Fig 1: schematic + cosine histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1, alpha=0.3)
    coords = project_pca_2d(x)
    ax.scatter(coords[:, 0], coords[:, 1], s=8, c=y.numpy(), alpha=0.5, cmap="nipy_spectral")
    ax.set_title(f"Embeddings on sphere (PCA 2D)\nuniformity loss = {unif_val:.3f} (t={args.uniformity_t})")
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    bins = np.linspace(-1, 1, 50)
    ax.hist(cos_rand, bins=bins, alpha=0.5, density=True, label=f"random sphere (L={unif_random:.2f})")
    ax.hist(cos_emb, bins=bins, alpha=0.5, density=True, label=f"encoder (L={unif_val:.2f})")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("pairwise cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("Pairwise cosines (off-diagonal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "uniformity_overview.png", dpi=160)
    plt.close(fig)

    # --- Fig 2: collapsed vs spread schematic (concept) ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, title, pts in zip(
        axes,
        ["Collapse (bad)", "Your embeddings", "Uniform on sphere (ideal)"],
        [
            np.array([[0.95, 0.1], [0.93, 0.15], [0.97, 0.05], [0.94, 0.12]]),
            coords[:80],
            np.column_stack([np.cos(theta), np.sin(theta)]),
        ],
    ):
        if title.startswith("Uniform"):
            ax.scatter(pts[:, 0], pts[:, 1], s=15, alpha=0.6)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=25 if title.startswith("Collapse") else 12, alpha=0.7)
        circle = plt.Circle((0, 0), 1, fill=False, color="gray", ls="--", lw=0.8)
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Uniformity = spread points on the hypersphere (avoid collapse)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "uniformity_concept.png", dpi=160)
    plt.close(fig)

    summary = {
        "uniformity_loss": unif_val,
        "uniformity_random_sphere": unif_random,
        "mean_pairwise_cosine": float(cos_emb.mean()),
        "random_mean_pairwise_cosine": float(cos_rand.mean()),
        "uniformity_t": args.uniformity_t,
        "n_samples": x.size(0),
        "note": "Lower uniformity loss = more spread. Compare to random on same sphere.",
    }
    import json

    (out / "uniformity_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved plots to {out}")


if __name__ == "__main__":
    main()
