#!/usr/bin/env python3
"""Extract encoder embeddings from a checkpoint and plot them in 2D.

Example:
  PYTHONPATH=. python scripts/visualize_embeddings_2d.py \\
    --checkpoint results/paper_cifar100/latent_infonce_jepa_augment/checkpoint_best.pt

  PYTHONPATH=. python scripts/visualize_embeddings_2d.py --auto-best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tripletjepa.data import get_dataloaders
from tripletjepa.eval import extract_features
from tripletjepa.models import TripletJEPA


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def find_best_checkpoints(root: Path) -> list[Path]:
    return sorted(root.glob("**/checkpoint_best.pt"))


def load_encoder_from_checkpoint(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    _, _, spec = get_dataloaders(
        cfg["dataset"],
        cfg.get("data_dir", "./data"),
        batch_size=256,
        num_workers=2,
        train_subset=cfg.get("train_subset"),
        download=False,
    )
    model = TripletJEPA(
        in_channels=spec.in_channels,
        embed_dim=cfg["embed_dim"],
        ema_momentum=cfg.get("ema_momentum", 0.996),
        use_ema_target=cfg.get("use_ema_target", True),
        num_prototypes=cfg.get("proto_num", 0) if cfg.get("proto_weight", 0) > 0 else 0,
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
        "training_mode": cfg.get("training_mode"),
        "dataset": cfg["dataset"],
        "num_classes": spec.num_classes,
        "checkpoint": str(checkpoint),
    }
    return model.encoder, meta


def subsample(x: torch.Tensor, y: torch.Tensor, max_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if x.size(0) <= max_samples:
        return x, y
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.size(0), generator=gen)[:max_samples]
    return x[idx], y[idx]


def project_2d(x: torch.Tensor, method: str, seed: int) -> np.ndarray:
    x_np = x.numpy()
    if method == "pca":
        x_c = x_np - x_np.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_c, full_matrices=False)
        return x_c @ vt[:2].T
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError("t-SNE requires scikit-learn: pip install scikit-learn") from exc
        perplexity = min(30, max(5, x_np.shape[0] // 50))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(x_np)
    raise ValueError(f"Unknown projection={method!r}")


def plot_2d(
    coords: np.ndarray,
    labels: torch.Tensor,
    title: str,
    out_path: Path,
    *,
    normalized: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels.numpy(),
        s=8,
        alpha=0.7,
        cmap="nipy_spectral",
    )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="class id")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D embedding visualization from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint_best.pt")
    parser.add_argument(
        "--auto-best",
        action="store_true",
        help="Use newest checkpoint_best.pt under --results-root",
    )
    parser.add_argument("--results-root", default="./results/paper_cifar100")
    parser.add_argument("--output-dir", default=None, help="Default: <run_dir>/embedding_viz")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--projection", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize before PCA/t-SNE")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-embeddings", action="store_true", help="Save .pt with features + labels")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.auto_best:
        candidates = find_best_checkpoints(Path(args.results_root))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint_best.pt under {args.results_root}")
        ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"auto-best: {ckpt_path}")
    else:
        parser.error("Provide --checkpoint or --auto-best")

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    encoder, meta = load_encoder_from_checkpoint(ckpt_path, device)

    train_loader, test_loader, _ = get_dataloaders(
        meta["dataset"],
        "./data",
        batch_size=256,
        num_workers=2,
        download=False,
    )
    loader = test_loader if args.split == "test" else train_loader
    print(f"Extracting {args.split} embeddings from {meta['run_name']} (epoch {meta['epoch']})...")
    features, labels = extract_features(encoder, loader, device)

    if args.normalize:
        features = torch.nn.functional.normalize(features, dim=-1)

    viz_x, viz_y = subsample(features, labels, args.max_samples, args.seed)
    coords = project_2d(viz_x, args.projection, args.seed)

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "embedding_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_tag = "norm" if args.normalize else "raw"
    png_path = out_dir / f"embedding_{args.split}_{args.projection}_{norm_tag}.png"
    title = (
        f"{meta['run_name']} | {args.split} | {args.projection.upper()} | "
        f"n={viz_x.size(0)} | epoch={meta['epoch']}"
    )
    plot_2d(coords, viz_y, title, png_path, normalized=args.normalize)

    summary = {
        **meta,
        "split": args.split,
        "projection": args.projection,
        "normalize": args.normalize,
        "num_points": int(viz_x.size(0)),
        "embed_dim": int(features.size(1)),
        "plot": str(png_path),
    }
    with (out_dir / "viz_meta.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if args.save_embeddings:
        emb_path = out_dir / f"embeddings_{args.split}.pt"
        torch.save({"features": features, "labels": labels, "meta": summary}, emb_path)
        print(f"Saved embeddings: {emb_path}")

    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
