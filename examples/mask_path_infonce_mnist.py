#!/usr/bin/env python3
"""Minimal mask-path InfoNCE demo (no EMA).

Matches the paper's placement: InfoNCE on predictor(mask) vs stop-grad encoder(clean).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def infonce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    q = F.normalize(query, dim=-1)
    k = F.normalize(key.detach(), dim=-1)
    logits = q @ k.T / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


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


class MaskPathJEPA(nn.Module):
    """Single encoder + mask-path predictor. No EMA teacher."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.encoder = SmallEncoder(embed_dim)
        self.predictor = Predictor(embed_dim)

    def forward(self, masked: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_hat = self.predictor(self.encoder(masked))
        z_clean = self.encoder(clean)
        return z_hat, z_clean


def block_mask(x: torch.Tensor, mask_ratio: float = 0.6) -> torch.Tensor:
    b, _, h, w = x.shape
    out = x.clone()
    patch = max(h // 4, 1)
    n_masks = int(mask_ratio * (h // patch) * (w // patch))
    for i in range(b):
        for _ in range(n_masks):
            top = torch.randint(0, h - patch + 1, (1,)).item()
            left = torch.randint(0, w - patch + 1, (1,)).item()
            out[i, :, top : top + patch, left : left + patch] = 0.0
    return out


def train_one_epoch(
    model: MaskPathJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for images, _ in loader:
        images = images.to(device)
        z_hat, z_clean = model(block_mask(images), images)
        loss = infonce_loss(z_hat, z_clean, temperature=temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = images.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def knn_accuracy(
    model: MaskPathJEPA,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    max_train: int = 2000,
) -> float:
    model.eval()
    train_x, train_y = [], []
    for images, labels in train_loader:
        train_x.append(model.encoder(images.to(device)).cpu())
        train_y.append(labels)
        if sum(t.size(0) for t in train_x) >= max_train:
            break
    train_x = F.normalize(torch.cat(train_x), dim=-1)
    train_y = torch.cat(train_y)

    correct = total = 0
    for images, labels in test_loader:
        q = F.normalize(model.encoder(images.to(device)), dim=-1).cpu()
        nn_idx = (q @ train_x.T).topk(5, dim=-1).indices
        pred = torch.mode(train_y[nn_idx], dim=-1).values
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal mask-path InfoNCE on MNIST")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
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

    model = MaskPathJEPA().to(device)
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
    )

    print("Training mask-path InfoNCE (no EMA)")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, args.temperature)
        knn = knn_accuracy(model, train_loader, test_loader, device)
        print(f"epoch {epoch:02d} | infonce {loss:.4f} | 5-NN {knn:.3f}")


if __name__ == "__main__":
    main()
