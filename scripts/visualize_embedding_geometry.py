#!/usr/bin/env python3
"""Visualize real checkpoint geometry: class PCA + k-NN neighborhoods (toy-style).

Example:
  PYTHONPATH=. python scripts/visualize_embedding_geometry.py \\
    --checkpoint results/paper_cifar100/latent_infonce_jepa_augment/checkpoint_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tripletjepa.data import get_dataloaders
from tripletjepa.eval import extract_features, knn_predict
from tripletjepa.models import TripletJEPA


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def load_encoder(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    _, _, spec = get_dataloaders(
        cfg["dataset"],
        cfg.get("data_dir", "./data"),
        batch_size=256,
        num_workers=2,
        download=False,
    )
    model = TripletJEPA(
        in_channels=spec.in_channels,
        embed_dim=cfg["embed_dim"],
        ema_momentum=cfg.get("ema_momentum", 0.996),
        use_ema_target=cfg.get("use_ema_target", True),
        num_prototypes=0,
        backbone=cfg.get("backbone", "resnet"),
        image_size=spec.image_size,
        vit_patch_size=cfg.get("vit_patch_size", 4),
        vit_depth=cfg.get("vit_depth", 6),
        vit_heads=cfg.get("vit_heads", 4),
        vit_mlp_dim=cfg.get("vit_mlp_dim", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {
        "epoch": ckpt.get("epoch"),
        "run_name": cfg.get("run_name", checkpoint.parent.name),
        "knn_k": cfg.get("knn_k", 20),
        "num_classes": spec.num_classes,
    }
    return model.encoder, meta


def pca_2d(x: torch.Tensor) -> np.ndarray:
    x_np = x.numpy()
    x_c = x_np - x_np.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_c, full_matrices=False)
    return x_c @ vt[:2].T


def plot_class_scatter(
    coords: np.ndarray,
    labels: torch.Tensor,
    title: str,
    path: Path,
    *,
    alpha: float = 0.35,
    s: float = 4,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels.numpy(), s=s, alpha=alpha, cmap="nipy_spectral")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="class")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def knn_neighbor_indices(
    train_x: torch.Tensor,
    query_x: torch.Tensor,
    k: int,
) -> torch.Tensor:
    train_n = F.normalize(train_x, dim=-1)
    query_n = F.normalize(query_x, dim=-1)
    sim = query_n @ train_n.T
    return sim.topk(k, dim=-1).indices


def plot_knn_panels(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    train_y: torch.Tensor,
    test_y: torch.Tensor,
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    *,
    k: int,
    n_panels: int,
    seed: int,
    title: str,
    path: Path,
) -> None:
    rng = np.random.default_rng(seed)
    n_panels = min(n_panels, test_x.size(0))
    pick = rng.choice(test_x.size(0), size=n_panels, replace=False)

    cols = min(2, n_panels)
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    nn_idx_all = knn_neighbor_indices(train_x, test_x[pick], k)

    for ax_i, (ti, local_i) in enumerate(zip(pick, range(n_panels))):
        ax = axes[local_i]
        ax.scatter(
            train_coords[:, 0],
            train_coords[:, 1],
            c=train_y.numpy(),
            s=3,
            alpha=0.12,
            cmap="nipy_spectral",
        )
        qx, qy = test_coords[ti]
        true = int(test_y[ti].item())
        nn_idx = nn_idx_all[local_i]
        nn_labels = train_y[nn_idx]
        pred = torch.mode(nn_labels, dim=0).values.item()
        ok = pred == true

        ax.scatter([qx], [qy], c="black", s=120, marker="*", zorder=5, label="test")
        for j, idx in enumerate(nn_idx.tolist()):
            nx, ny = train_coords[idx]
            ax.plot([qx, nx], [qy, ny], color="black", alpha=0.25, lw=0.8, zorder=2)
            ax.scatter([nx], [ny], s=60, facecolors="none", edgecolors="black", linewidths=1.2, zorder=4)
        ax.scatter([qx], [qy], c="black", s=120, marker="*", zorder=5)
        vote = "✓" if ok else "✗"
        ax.set_title(f"test class {true} | k-NN → {pred} {vote} (k={k})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA + k-NN neighborhood plots from checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train", type=int, default=8000, help="Train points in background scatter")
    parser.add_argument("--max-test", type=int, default=2000, help="Test points for PCA background")
    parser.add_argument("--knn-panels", type=int, default=4, help="Number of test examples with neighbor lines")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "embedding_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    encoder, meta = load_encoder(ckpt_path, device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    train_loader, test_loader, _ = get_dataloaders(
        cfg["dataset"],
        cfg.get("data_dir", "./data"),
        batch_size=256,
        num_workers=2,
        download=False,
    )

    print("Extracting train/test embeddings...")
    train_x, train_y = extract_features(encoder, train_loader, device)
    test_x, test_y = extract_features(encoder, test_loader, device)

    gen = torch.Generator().manual_seed(args.seed)
    if train_x.size(0) > args.max_train:
        idx = torch.randperm(train_x.size(0), generator=gen)[: args.max_train]
        train_x_s, train_y_s = train_x[idx], train_y[idx]
    else:
        train_x_s, train_y_s = train_x, train_y

    if test_x.size(0) > args.max_test:
        idx = torch.randperm(test_x.size(0), generator=gen)[: args.max_test]
        test_x_s, test_y_s = test_x[idx], test_y[idx]
    else:
        test_x_s, test_y_s = test_x, test_y

    # PCA fit on combined cloud (visualization only — eval still in 256D)
    combined = torch.cat([train_x_s, test_x_s], dim=0)
    coords_all = pca_2d(combined)
    train_coords = coords_all[: train_x_s.size(0)]
    test_coords = coords_all[train_x_s.size(0) :]

    k = meta["knn_k"]
    pred = knn_predict(train_x, train_y, test_x, k=k)
    knn_acc = (pred == test_y).float().mean().item()

    title_base = f"{meta['run_name']} epoch={meta['epoch']} | k-NN={knn_acc:.3f} (full test)"

    pca_path = out_dir / "geometry_pca_by_class.png"
    plot_class_scatter(
        np.vstack([train_coords, test_coords]),
        torch.cat([train_y_s, test_y_s]),
        title_base + " | PCA colored by class",
        pca_path,
    )

    knn_path = out_dir / "geometry_knn_neighborhoods.png"
    # Pick test indices from full test set; map coords via separate PCA on full test for panels
    test_coords_full = pca_2d(test_x)
    plot_knn_panels(
        pca_2d(train_x),
        test_coords_full,
        train_y,
        test_y,
        train_x,
        test_x,
        k=k,
        n_panels=args.knn_panels,
        seed=args.seed,
        title=title_base + " | local k-NN neighborhoods (toy-style)",
        path=knn_path,
    )

    summary = {
        "checkpoint": str(ckpt_path),
        "knn_acc_full_test": knn_acc,
        "knn_k": k,
        "plots": [str(pca_path), str(knn_path)],
    }
    with (out_dir / "geometry_meta.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"k-NN on full test (recomputed): {knn_acc:.4f}")
    print(f"Saved: {pca_path}")
    print(f"Saved: {knn_path}")


if __name__ == "__main__":
    main()
