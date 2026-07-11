from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    feats, labels = [], []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        feats.append(encoder(images).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    k: int = 20,
) -> float:
    """k-NN classification with cosine distance and majority vote (DINO-style)."""
    train_x = F.normalize(train_x, dim=-1)
    test_x = F.normalize(test_x, dim=-1)
    sim = test_x @ train_x.T  # cosine sim when vectors are unit-normalized
    nn_idx = sim.topk(k, dim=-1).indices  # k nearest train indices per test image
    nn_labels = train_y[nn_idx]  # [num_test, k]
    # Majority vote: most common label among the k neighbors.
    pred = torch.mode(nn_labels, dim=-1).values
    return (pred == test_y).float().mean().item()


def linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    epochs: int = 50,
    lr: float = 0.1,
    device: torch.device | None = None,
) -> float:
    """Train a linear classifier on frozen features; report test accuracy."""
    device = device or torch.device("cpu")
    probe = nn.Linear(train_x.size(-1), num_classes).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    x_tr, y_tr = train_x.to(device), train_y.to(device)
    x_te, y_te = test_x.to(device), test_y.to(device)

    for _ in range(epochs):
        probe.train()
        logits = probe(x_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(x_te).argmax(dim=-1)
    return (pred == y_te).float().mean().item()


@torch.no_grad()
def evaluate_encoder(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    knn_k: int = 20,
    probe_epochs: int = 50,
) -> dict[str, float]:
    train_x, train_y = extract_features(encoder, train_loader, device)
    test_x, test_y = extract_features(encoder, test_loader, device)
    return {
        "knn": knn_accuracy(train_x, train_y, test_x, test_y, k=knn_k),
        "linear_probe": linear_probe(
            train_x, train_y, test_x, test_y, num_classes, epochs=probe_epochs, device=device
        ),
    }
