#!/usr/bin/env python3
"""Plot test queries and their five nearest CIFAR-10 training images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mask_path_jepa.data import DATASETS
from mask_path_jepa.models import JEPA


@torch.no_grad()
def encode(encoder: torch.nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    chunks = []
    encoder.eval()
    for images, _ in loader:
        chunks.append(encoder(images.to(device, non_blocking=True)).cpu())
    return torch.cat(chunks)


def load_encoder(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    spec = DATASETS[cfg["dataset"]]
    model = JEPA(
        in_channels=spec.in_channels,
        embed_dim=cfg["embed_dim"],
        ema_momentum=cfg.get("ema_momentum", 0.996),
        use_ema_target=cfg.get("use_ema_target", False),
        num_prototypes=0,
        backbone=cfg.get("backbone", "resnet18"),
        image_size=spec.image_size,
        vit_patch_size=cfg.get("vit_patch_size", 4),
        vit_depth=cfg.get("vit_depth", 6),
        vit_heads=cfg.get("vit_heads", 4),
        vit_mlp_dim=cfg.get("vit_mlp_dim", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model.encoder, {"epoch": ckpt.get("epoch"), "config": cfg}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--query-indices", type=int, nargs="+", default=None,
                        help="Explicit test indices; overrides --queries and random sampling")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, meta = load_encoder(checkpoint, device)
    cfg = meta["config"]
    if cfg["dataset"] != "cifar10":
        raise ValueError("This figure currently supports CIFAR-10 checkpoints only")

    spec = DATASETS["cifar10"]
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(spec.mean, spec.std),
    ])
    train_eval = datasets.CIFAR10(cfg.get("data_dir", "./data"), train=True,
                                  transform=eval_transform, download=False)
    test_eval = datasets.CIFAR10(cfg.get("data_dir", "./data"), train=False,
                                 transform=eval_transform, download=False)
    train_raw = datasets.CIFAR10(cfg.get("data_dir", "./data"), train=True,
                                 transform=None, download=False)
    test_raw = datasets.CIFAR10(cfg.get("data_dir", "./data"), train=False,
                                transform=None, download=False)
    train_loader = DataLoader(train_eval, batch_size=512, shuffle=False, num_workers=2,
                              pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_eval, batch_size=512, shuffle=False, num_workers=2,
                             pin_memory=torch.cuda.is_available())

    print("Encoding deterministic CIFAR-10 train and test splits...")
    train_z = F.normalize(encode(encoder, train_loader, device), dim=-1)
    test_z = F.normalize(encode(encoder, test_loader, device), dim=-1)
    generator = torch.Generator().manual_seed(args.seed)
    if args.query_indices is not None:
        query_idx = torch.tensor(args.query_indices, dtype=torch.long)
    else:
        query_idx = torch.randperm(len(test_eval), generator=generator)[:args.queries]
    n_queries = len(query_idx)
    similarity = test_z[query_idx] @ train_z.T
    values, neighbor_idx = similarity.topk(args.neighbors, dim=1)

    cols = args.neighbors + 1
    fig, axes = plt.subplots(n_queries, cols, figsize=(2.05 * cols, 1.95 * n_queries))
    if n_queries == 1:
        axes = axes[None, :]
    records = []
    for row, qi in enumerate(query_idx.tolist()):
        query_image, query_label = test_raw[qi]
        ax = axes[row, 0]
        ax.imshow(query_image)
        ax.set_title(f"Query: {test_raw.classes[query_label]}", fontsize=9, fontweight="bold")
        ax.add_patch(Rectangle((0, 0), 31, 31, fill=False, edgecolor="black", linewidth=2.5))
        ax.axis("off")
        item = {"query_index": qi, "query_label": test_raw.classes[query_label], "neighbors": []}
        for rank, (ni, sim) in enumerate(zip(neighbor_idx[row].tolist(), values[row].tolist()), start=1):
            image, label = train_raw[ni]
            ax = axes[row, rank]
            ax.imshow(image)
            match = label == query_label
            color = "#238b45" if match else "#cb181d"
            ax.add_patch(Rectangle((0, 0), 31, 31, fill=False, edgecolor=color, linewidth=2.5))
            ax.set_title(f"{rank}. {train_raw.classes[label]}\ncos={sim:.3f}", fontsize=8)
            ax.axis("off")
            item["neighbors"].append({"rank": rank, "train_index": ni,
                                      "label": train_raw.classes[label], "cosine": sim})
        records.append(item)

    fig.suptitle("Five nearest training images in the learned embedding space", fontsize=12, y=0.995)
    fig.text(0.5, 0.004, "Green border: same class as query; red border: different class",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.975), h_pad=1.0, w_pad=0.4)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "checkpoint": str(checkpoint), "epoch": meta["epoch"], "seed": args.seed,
        "neighbors": args.neighbors, "metric": "cosine", "queries": records,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {output} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
