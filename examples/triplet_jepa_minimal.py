#!/usr/bin/env python3
"""
Minimal Triplet-JEPA example (educational).

Combines:
  - JEPA regression: predictor(corrupt) ≈ target_encoder(clean)
  - Triplet margin:   d(anchor, pos) + margin < d(anchor, neg)

See ssl_jepa_triplet/NOTES.md for research context.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def jepa_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, averaged over batch."""
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """FaceNet-style margin ranking on L2 distance (embeddings need not be normalized)."""
    d_pos = (anchor - positive).pow(2).sum(dim=-1)
    d_neg = (anchor - negative).pow(2).sum(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


@dataclass
class TripletJEPALoss:
    margin: float = 0.2
    triplet_weight: float = 0.5

    def __call__(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_jepa = jepa_cosine_loss(z_anchor, z_positive)
        l_triplet = triplet_loss(z_anchor, z_positive, z_negative, margin=self.margin)
        total = l_jepa + self.triplet_weight * l_triplet
        return total, {
            "loss": total.item(),
            "jepa": l_jepa.item(),
            "triplet": l_triplet.item(),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SmallEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TripletJEPA(nn.Module):
    """Context encoder + predictor + EMA target encoder."""

    def __init__(self, embed_dim: int = 128, ema_momentum: float = 0.99) -> None:
        super().__init__()
        self.encoder = SmallEncoder(embed_dim)
        self.predictor = Predictor(embed_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        m = self.ema_momentum
        for online, target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(m).add_(online.data, alpha=1.0 - m)

    def forward(self, corrupt: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_anchor = self.predictor(self.encoder(corrupt))
        z_positive = self.target_encoder(clean)
        return z_anchor, z_positive


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def corrupt_view(x: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """Simple JEPA-style corruption: zero random square patches."""
    b, c, h, w = x.shape
    out = x.clone()
    patch = max(h // 4, 1)
    n_masks = int(mask_ratio * (h // patch) * (w // patch))
    for i in range(b):
        for _ in range(n_masks):
            top = torch.randint(0, h - patch + 1, (1,)).item()
            left = torch.randint(0, w - patch + 1, (1,)).item()
            out[i, :, top : top + patch, left : left + patch] = 0.0
    return out


def batch_negatives(z: torch.Tensor) -> torch.Tensor:
    """Instance negatives via batch shuffle (unsupervised)."""
    idx = torch.randperm(z.size(0), device=z.device)
    return z[idx]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: TripletJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: TripletJEPALoss,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "jepa": 0.0, "triplet": 0.0}
    n = 0

    for images, _ in loader:
        images = images.to(device)
        corrupt = corrupt_view(images)
        clean = images

        z_anchor, z_positive = model(corrupt, clean)
        z_negative = batch_negatives(z_positive.detach())

        loss, stats = criterion(z_anchor, z_positive.detach(), z_negative)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_target_encoder()

        bs = images.size(0)
        n += bs
        for k in totals:
            totals[k] += stats[k] * bs

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def knn_accuracy(
    model: TripletJEPA,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    max_train: int = 2000,
) -> float:
    """Quick geometry check: 5-NN on frozen encoder embeddings."""
    model.eval()
    train_x, train_y = [], []
    for images, labels in train_loader:
        images = images.to(device)
        train_x.append(model.encoder(images).cpu())
        train_y.append(labels)
        if sum(t.size(0) for t in train_x) >= max_train:
            break
    train_x = F.normalize(torch.cat(train_x), dim=-1)
    train_y = torch.cat(train_y)

    correct = total = 0
    for images, labels in test_loader:
        images = images.to(device)
        q = F.normalize(model.encoder(images), dim=-1).cpu()
        sim = q @ train_x.T
        nn_idx = sim.topk(5, dim=-1).indices
        nn_labels = train_y[nn_idx]
        pred = torch.mode(nn_labels, dim=-1).values
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Triplet-JEPA on MNIST")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--triplet-weight", type=float, default=0.5)
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = TripletJEPA().to(device)
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
    )
    criterion = TripletJEPALoss(margin=args.margin, triplet_weight=args.triplet_weight)

    print("Training Triplet-JEPA (JEPA regression + batch triplet negatives)")
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        knn = knn_accuracy(model, train_loader, test_loader, device)
        print(
            f"epoch {epoch:02d} | loss {stats['loss']:.4f} "
            f"(jepa {stats['jepa']:.4f}, triplet {stats['triplet']:.4f}) | 5-NN {knn:.3f}"
        )


if __name__ == "__main__":
    main()
