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
    for batch in loader:
        if len(batch) == 3:
            images, y, _ = batch
        else:
            images, y = batch
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
def knn_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    k: int = 20,
) -> torch.Tensor:
    """Per-test k-NN predictions with cosine similarity."""
    train_x = F.normalize(train_x, dim=-1)
    test_x = F.normalize(test_x, dim=-1)
    sim = test_x @ train_x.T
    nn_idx = sim.topk(k, dim=-1).indices
    nn_labels = train_y[nn_idx]
    return torch.mode(nn_labels, dim=-1).values


def linear_probe_fit_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    epochs: int = 50,
    lr: float = 0.1,
    device: torch.device | None = None,
) -> tuple[float, torch.Tensor, nn.Linear]:
    """Train linear probe; return accuracy, test predictions, fitted probe."""
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
    acc = (pred == y_te).float().mean().item()
    return acc, pred.cpu(), probe


@torch.no_grad()
def feature_spectrum(x: torch.Tensor) -> dict[str, torch.Tensor | float]:
    """Singular values and rank stats for raw feature matrix (N, D)."""
    x = x - x.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(x, full_matrices=False)
    s2 = s.square()
    total = s2.sum().clamp(min=1e-12)
    cumvar = torch.cumsum(s2, dim=0) / total
    eff_rank = (s.sum().square() / s2.sum().clamp(min=1e-12)).item()
    dims_90 = int((cumvar >= 0.90).nonzero(as_tuple=False)[0].item()) + 1
    dims_99 = int((cumvar >= 0.99).nonzero(as_tuple=False)[0].item()) + 1
    return {
        "singular_values": s,
        "cumvar": cumvar,
        "effective_rank": eff_rank,
        "dims_90": dims_90,
        "dims_99": dims_99,
    }


@torch.no_grad()
def within_between_class_cosine(
    x: torch.Tensor,
    y: torch.Tensor,
    max_pairs: int = 50_000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean cosine similarity within vs between classes (on normalized features)."""
    z = F.normalize(x, dim=-1)
    n = z.size(0)
    if n < 2:
        return {"within": float("nan"), "between": float("nan"), "gap": float("nan")}

    gen = torch.Generator()
    gen.manual_seed(seed)
    num_pairs = min(max_pairs, n * (n - 1) // 2)
    idx_a = torch.randint(0, n, (num_pairs,), generator=gen)
    idx_b = torch.randint(0, n, (num_pairs,), generator=gen)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    if idx_a.numel() == 0:
        return {"within": float("nan"), "between": float("nan"), "gap": float("nan")}

    sim = (z[idx_a] * z[idx_b]).sum(dim=-1)
    same = y[idx_a] == y[idx_b]
    within = sim[same].mean().item() if same.any() else float("nan")
    between = sim[~same].mean().item() if (~same).any() else float("nan")
    gap = within - between if same.any() and (~same).any() else float("nan")
    return {"within": within, "between": between, "gap": gap}


def per_class_accuracy(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-class accuracy vector (NaN if class absent in target)."""
    acc = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        mask = target == c
        if mask.any():
            acc[c] = (pred[mask] == target[mask]).float().mean()
    return acc


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
        # linear_probe needs autograd; must not run inside @torch.no_grad().
        "linear_probe": linear_probe(
            train_x, train_y, test_x, test_y, num_classes, epochs=probe_epochs, device=device
        ),
    }
