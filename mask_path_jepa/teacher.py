"""Frozen pretrained teacher (DINOv2) and precomputed feature bank."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dinov2(name: str = "dinov2_vits14", device: torch.device | None = None) -> nn.Module:
    """Load frozen DINOv2 from torch hub (ViT-S/14 ~22M params)."""
    device = device or torch.device("cpu")
    model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


@torch.no_grad()
def encode_teacher_batch(
    model: nn.Module,
    images: torch.Tensor,
    *,
    pool: str = "cls",
) -> torch.Tensor:
    """Encode a batch already resized & ImageNet-normalized for DINOv2."""
    out = model.forward_features(images)
    if pool == "cls":
        return out["x_norm_clstoken"]
    if pool == "mean_patch":
        return out["x_norm_patchtokens"].mean(dim=1)
    raise ValueError(f"Unknown pool={pool!r}; expected cls or mean_patch")


def teacher_weighted_align_loss(
    z: torch.Tensor,
    phi: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pull encoder embeddings together when frozen teacher features agree."""
    z = F.normalize(z, dim=-1)
    phi = F.normalize(phi, dim=-1)
    b = z.size(0)
    if b < 2:
        return z.new_zeros(())
    logits = phi @ phi.T / temperature
    logits = logits.masked_fill(torch.eye(b, device=z.device, dtype=torch.bool), float("-inf"))
    weights = F.softmax(logits, dim=1)
    sim_z = z @ z.T
    return (weights * (1.0 - sim_z)).sum(dim=1).mean()


class TeacherFeatureBank:
    """Lookup of precomputed teacher features by dataset index."""

    def __init__(self, path: str | Path, device: torch.device = torch.device("cpu")) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Teacher features not found: {path}\n"
                "Run: PYTHONPATH=. python scripts/extract_teacher_features.py --dataset ..."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.features = payload["features"].to(device)
        self.meta = {k: v for k, v in payload.items() if k != "features"}
        self.device = device

    def __len__(self) -> int:
        return self.features.size(0)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.detach().cpu().long()
        return self.features[idx].to(indices.device)

    def describe(self) -> str:
        return (
            f"teacher={self.meta.get('teacher')} "
            f"pool={self.meta.get('pool')} "
            f"dim={self.features.size(1)} "
            f"n={len(self)} "
            f"image_size={self.meta.get('image_size')}"
        )


def teacher_weight_at_epoch(
    epoch: int,
    weight: float,
    anneal_start: int,
    anneal_end: int,
) -> float:
    if weight <= 0 or epoch >= anneal_end:
        return 0.0
    if epoch <= anneal_start:
        return weight
    if anneal_end <= anneal_start:
        return weight
    frac = (anneal_end - epoch) / (anneal_end - anneal_start)
    return weight * max(0.0, min(1.0, frac))
