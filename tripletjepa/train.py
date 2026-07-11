from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tripletjepa.data import DatasetSpec, get_dataloaders
from tripletjepa.eval import evaluate_encoder
from tripletjepa.losses import TripletJEPALoss
from tripletjepa.models import TripletJEPA
from tripletjepa.views import block_mask, class_negatives, instance_negatives


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    # Data
    dataset: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    train_subset: int | None = None
    download: bool = False

    # Model
    embed_dim: int = 256
    ema_momentum: float = 0.996

    # Loss
    margin: float = 0.2
    triplet_weight: float = 0.5
    negative_mode: str = "instance"  # instance | class
    mask_ratio: float = 0.6

    # Optim
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Eval
    eval_every: int = 10
    knn_k: int = 20
    probe_epochs: int = 50

    # Run
    seed: int = 42
    device: str = "auto"
    output_dir: str = "./results/run"
    run_name: str = "triplet_jepa"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        data = json.loads(Path(path).read_text())
        known = {f.name for f in cls.__dataclass_fields__.values()}
        core = {k: v for k, v in data.items() if k in known and k != "extra"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**core, extra=extra)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def train_one_epoch(
    model: TripletJEPA,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: TripletJEPALoss,
    device: torch.device,
    spec: DatasetSpec,
    cfg: TrainConfig,
) -> dict[str, float]:
    """One SSL epoch: corrupt view → predict clean latent + optional triplet margin."""
    model.train()
    totals = {"loss": 0.0, "jepa": 0.0, "triplet": 0.0}
    n = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Context view: masked image. Target view: clean (augmented) image.
        corrupt = block_mask(images, mask_ratio=cfg.mask_ratio)

        # ẑ = predictor(encoder(corrupt)),  z+ = target_encoder(clean)
        z_anchor, z_positive = model(corrupt, images)
        z_pos = z_positive.detach()  # stop-grad on target path (I-JEPA style)

        # z- = embedding of a different image (batch negative or different class).
        if cfg.triplet_weight > 0:
            if cfg.negative_mode == "class":
                z_neg = class_negatives(z_pos, labels)
            else:
                z_neg = instance_negatives(z_pos)
        else:
            # λ=0 baseline: triplet term is zeroed by weight, but loss fn still expects z-.
            z_neg = instance_negatives(z_pos)

        loss, stats = criterion(z_anchor, z_pos, z_neg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_target_encoder()  # EMA: target_encoder ← m·target + (1-m)·encoder

        bs = images.size(0)
        n += bs
        for k in totals:
            totals[k] += stats[k] * bs

    return {k: v / max(n, 1) for k, v in totals.items()}


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def run_training(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    out = Path(cfg.output_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")

    train_loader, test_loader, spec = get_dataloaders(
        cfg.dataset,
        cfg.data_dir,
        cfg.batch_size,
        cfg.num_workers,
        cfg.train_subset,
        download=cfg.download,
    )

    model = TripletJEPA(
        in_channels=spec.in_channels,
        embed_dim=cfg.embed_dim,
        ema_momentum=cfg.ema_momentum,
    ).to(device)
    params = model.param_count()

    optimizer = optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = TripletJEPALoss(margin=cfg.margin, triplet_weight=cfg.triplet_weight)

    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, spec, cfg
        )
        scheduler.step()
        row: dict = {"epoch": epoch, **train_stats, "lr": scheduler.get_last_lr()[0]}

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            # Paper metrics: frozen encoder only (predictor is not used at eval).
            metrics = evaluate_encoder(
                model.encoder,
                train_loader,
                test_loader,
                spec.num_classes,
                device,
                knn_k=cfg.knn_k,
                probe_epochs=cfg.probe_epochs,
            )
            row.update(metrics)

        history.append(row)
        tag = (
            f"epoch {epoch:03d}/{cfg.epochs} | loss {row['loss']:.4f} "
            f"(jepa {row['jepa']:.4f}, triplet {row['triplet']:.4f})"
        )
        if "knn" in row:
            tag += f" | k-NN {row['knn']:.3f} | linear {row['linear_probe']:.3f}"
        print(tag)

    elapsed = time.time() - t0
    # Save encoder weights for downstream eval / deployment (inference artifact).
    torch.save(model.encoder.state_dict(), out / "encoder.pt")

    summary = {
        "run_name": cfg.run_name,
        "dataset": cfg.dataset,
        "triplet_weight": cfg.triplet_weight,
        "negative_mode": cfg.negative_mode,
        "margin": cfg.margin,
        "epochs": cfg.epochs,
        "params": params,
        "elapsed_sec": round(elapsed, 1),
        "device": str(device),
        "final": history[-1],
        "history": history,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    return summary
