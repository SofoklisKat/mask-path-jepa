#!/usr/bin/env python3
"""Visualize SSL embeddings and run k-NN vs linear error analysis.

Example (InfoNCE only):
  PYTHONPATH=. python scripts/analyze_embeddings.py \\
    --checkpoint results/paper_cifar100/latent_infonce_jepa_augment/checkpoint_best.pt \\
    --name infonce \\
    --output-dir results/analysis/infonce
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mask_path_jepa.data import get_dataloaders
from mask_path_jepa.eval import (
    extract_features,
    feature_spectrum,
    knn_predict,
    linear_probe_fit_predict,
    per_class_accuracy,
    within_between_class_cosine,
)
from mask_path_jepa.models import JEPA


def load_encoder(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    train_loader, _, spec = get_dataloaders(
        cfg["dataset"],
        cfg.get("data_dir", "./data"),
        batch_size=256,
        num_workers=2,
        train_subset=cfg.get("train_subset"),
        download=False,
    )
    del train_loader

    model = JEPA(
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
        "knn_k": cfg.get("knn_k", 20),
        "probe_epochs": cfg.get("probe_epochs", 50),
    }
    return model.encoder, meta


def subsample(
    x: torch.Tensor,
    y: torch.Tensor,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.size(0) <= max_samples:
        return x, y
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.size(0), generator=gen)[:max_samples]
    return x[idx], y[idx]


def project_2d(
    x: torch.Tensor,
    method: str,
    seed: int,
) -> np.ndarray:
    x_np = x.numpy()
    if method == "pca":
        x_c = x_np - x_np.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_c, full_matrices=False)
        return x_c @ vt[:2].T
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError(
                "t-SNE requires scikit-learn: pip install scikit-learn"
            ) from exc
        return TSNE(
            n_components=2,
            perplexity=min(30, max(5, x_np.shape[0] // 50)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(x_np)
    raise ValueError(f"Unknown projection method={method!r}")


def plot_projection(
    coords: np.ndarray,
    labels: torch.Tensor,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels.numpy(),
        s=6,
        alpha=0.65,
        cmap="tab20" if labels.max() < 20 else "nipy_spectral",
    )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="class")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_spectrum(
    singular_values: torch.Tensor,
    cumvar: torch.Tensor,
    title: str,
    path: Path,
) -> None:
    sv = singular_values.numpy()
    cv = cumvar.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(sv, lw=1.2)
    axes[0].set_title("Singular values")
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("σ")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cv, lw=1.2, color="darkorange")
    axes[1].axhline(0.9, color="gray", ls="--", lw=0.8)
    axes[1].axhline(0.99, color="gray", ls=":", lw=0.8)
    axes[1].set_title("Cumulative variance (centered features)")
    axes[1].set_xlabel("dimension")
    axes[1].set_ylabel("fraction")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_per_class_gap(
    knn_acc: torch.Tensor,
    lin_acc: torch.Tensor,
    title: str,
    path: Path,
    top_k: int = 20,
) -> None:
    gap = knn_acc - lin_acc
    valid = torch.isfinite(gap)
    gap = gap[valid]
    classes = torch.arange(knn_acc.numel())[valid]
    order = torch.argsort(gap, descending=True)
    top = order[:top_k]
    bottom = order[-top_k:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, idx, subtitle in (
        (axes[0], top, f"top {top_k}: k-NN >> linear"),
        (axes[1], bottom, f"bottom {top_k}: linear closer to k-NN"),
    ):
        cls = classes[idx].numpy()
        x = np.arange(len(cls))
        w = 0.35
        ax.bar(x - w / 2, knn_acc[valid][idx].numpy(), width=w, label="k-NN")
        ax.bar(x + w / 2, lin_acc[valid][idx].numpy(), width=w, label="linear")
        ax.set_xticks(x)
        ax.set_xticklabels(cls, rotation=90, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(subtitle)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_disagreement_pie(counts: dict[str, int], title: str, path: Path) -> None:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def top_confused_pairs(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    top_k: int = 20,
) -> list[tuple[int, int, int]]:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(target, pred):
        cm[t, p] += 1
    pairs: list[tuple[int, int, int]] = []
    for t in range(num_classes):
        for p in range(num_classes):
            if t != p and cm[t, p] > 0:
                pairs.append((int(t), int(p), int(cm[t, p].item())))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def analyze_one(
    name: str,
    encoder: torch.nn.Module,
    meta: dict,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    knn_pred = knn_predict(train_x, train_y, test_x, k=meta["knn_k"])
    lin_acc, lin_pred, _ = linear_probe_fit_predict(
        train_x,
        train_y,
        test_x,
        test_y,
        meta["num_classes"],
        epochs=meta["probe_epochs"],
        device=device,
    )
    knn_acc = (knn_pred == test_y).float().mean().item()

    knn_pc = per_class_accuracy(knn_pred, test_y, meta["num_classes"])
    lin_pc = per_class_accuracy(lin_pred, test_y, meta["num_classes"])

    both_ok = (knn_pred == test_y) & (lin_pred == test_y)
    knn_only = (knn_pred == test_y) & (lin_pred != test_y)
    lin_only = (knn_pred != test_y) & (lin_pred == test_y)
    both_bad = (knn_pred != test_y) & (lin_pred != test_y)
    disagree = knn_pred != lin_pred

    counts = {
        "both correct": int(both_ok.sum()),
        "k-NN only": int(knn_only.sum()),
        "linear only": int(lin_only.sum()),
        "both wrong": int(both_bad.sum()),
    }

    spec_train = feature_spectrum(train_x)
    spec_test = feature_spectrum(test_x)
    wb_train = within_between_class_cosine(train_x, train_y, seed=args.seed)
    wb_test = within_between_class_cosine(test_x, test_y, seed=args.seed)

    # Normalized vs raw linear probe (diagnostic for sphere-trained models)
    train_n = F.normalize(train_x, dim=-1)
    test_n = F.normalize(test_x, dim=-1)
    _, lin_pred_norm, _ = linear_probe_fit_predict(
        train_n,
        train_y,
        test_n,
        test_y,
        meta["num_classes"],
        epochs=meta["probe_epochs"],
        device=device,
    )
    lin_norm_acc = (lin_pred_norm == test_y).float().mean().item()

    summary = {
        "name": name,
        "run_name": meta["run_name"],
        "training_mode": meta["training_mode"],
        "epoch": meta["epoch"],
        "knn_acc": knn_acc,
        "linear_acc_raw": lin_acc,
        "linear_acc_normalized": lin_norm_acc,
        "within_cosine_train": wb_train["within"],
        "between_cosine_train": wb_train["between"],
        "class_gap_train": wb_train["gap"],
        "effective_rank_train": spec_train["effective_rank"],
        "dims_90_train": spec_train["dims_90"],
        "dims_99_train": spec_train["dims_99"],
        "effective_rank_test": spec_test["effective_rank"],
        "disagreement_counts": counts,
        "disagreement_rate": disagree.float().mean().item(),
    }

    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (run_dir / "per_class_accuracy.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "knn_acc", "linear_acc", "knn_minus_linear"])
        for c in range(meta["num_classes"]):
            if torch.isfinite(knn_pc[c]):
                writer.writerow(
                    [
                        c,
                        f"{knn_pc[c].item():.4f}",
                        f"{lin_pc[c].item():.4f}",
                        f"{(knn_pc[c] - lin_pc[c]).item():.4f}",
                    ]
                )

    with (run_dir / "linear_confusions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", "pred_class", "count"])
        for t, p, n in top_confused_pairs(lin_pred, test_y, meta["num_classes"]):
            writer.writerow([t, p, n])

    plot_spectrum(
        spec_train["singular_values"],
        spec_train["cumvar"],
        f"{name}: feature spectrum (train)",
        run_dir / "spectrum_train.png",
    )
    plot_per_class_gap(
        knn_pc,
        lin_pc,
        f"{name}: per-class k-NN vs linear",
        run_dir / "per_class_gap.png",
    )
    plot_disagreement_pie(
        counts,
        f"{name}: test prediction agreement",
        run_dir / "disagreement.png",
    )

    viz_x, viz_y = subsample(test_x, test_y, args.max_samples, args.seed)
    if args.projection != "none":
        coords = project_2d(viz_x, args.projection, args.seed)
        plot_projection(
            coords,
            viz_y,
            f"{name}: {args.projection.upper()} (test, n={viz_x.size(0)})",
            run_dir / f"embedding_{args.projection}.png",
        )

    print(f"\n=== {name} ({meta['run_name']}) ===")
    print(f"k-NN {knn_acc:.3f} | linear(raw) {lin_acc:.3f} | linear(normalized) {lin_norm_acc:.3f}")
    print(
        f"rank: eff={spec_train['effective_rank']:.1f} | "
        f"90% var in {spec_train['dims_90']} dims | "
        f"99% var in {spec_train['dims_99']} dims"
    )
    print(
        f"cosine train: within={wb_train['within']:.3f} "
        f"between={wb_train['between']:.3f} gap={wb_train['gap']:.3f}"
    )
    print(
        f"disagreement: k-NN-only={counts['k-NN only']} "
        f"linear-only={counts['linear only']} "
        f"both-wrong={counts['both wrong']}"
    )
    if lin_norm_acc > lin_acc + 0.02:
        print("hint: linear on NORMALIZED features is much better -> geometry is spherical; raw scale hurts linear probe")

    return summary


def compare_summaries(summaries: list[dict], out_dir: Path) -> None:
    if len(summaries) < 2:
        return
    names = [s["name"] for s in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = [
        ("knn_acc", "k-NN accuracy"),
        ("linear_acc_raw", "Linear (raw features)"),
        ("linear_acc_normalized", "Linear (normalized)"),
    ]
    for ax, (key, label) in zip(axes, metrics):
        vals = [s[key] for s in summaries]
        ax.bar(names, vals)
        ax.set_ylim(0, max(vals) * 1.15 if vals else 1)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Checkpoint comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_metrics.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding visualization and error analysis")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint_best.pt",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run label for plots (default: parent folder name)",
    )
    parser.add_argument("--output-dir", default="./results/analysis")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=5000, help="Points for 2D plot")
    parser.add_argument(
        "--projection",
        choices=["pca", "tsne", "none"],
        default="pca",
        help="2D embedding plot method",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = [Path(args.checkpoint)]
    names = [args.name or paths[0].parent.name]

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    if args.device != "auto":
        device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared data loaders from first checkpoint config
    first_ckpt = torch.load(paths[0], map_location="cpu", weights_only=False)
    cfg = first_ckpt["config"]
    train_loader, test_loader, spec = get_dataloaders(
        cfg["dataset"],
        cfg.get("data_dir", "./data"),
        batch_size=256,
        num_workers=2,
        train_subset=cfg.get("train_subset"),
        download=False,
    )

    summaries: list[dict] = []
    for path, name in zip(paths, names):
        encoder, meta = load_encoder(path, device)
        meta["num_classes"] = spec.num_classes
        train_x, train_y = extract_features(encoder, train_loader, device)
        test_x, test_y = extract_features(encoder, test_loader, device)
        summaries.append(
            analyze_one(name, encoder, meta, train_x, train_y, test_x, test_y, out_dir, args, device)
        )

    compare_summaries(summaries, out_dir)
    with (out_dir / "all_summaries.json").open("w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
