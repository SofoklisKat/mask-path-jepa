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
from tripletjepa.views import (
    active_patch_blur_ratio,
    class_negatives,
    corrupt_progress,
    instance_negatives,
    make_corrupt_view,
    scramble_patches,
)


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
    training_mode: str = "jepa_ema"  # jepa_ema | latent_triplet | latent_vicreg
    use_ema_target: bool = True
    anchor_mode: str = "predictor_corrupt"  # predictor_corrupt | encoder_corrupt | encoder_clean

    # Loss
    margin: float = 0.2
    triplet_weight: float = 0.5
    negative_mode: str = "instance"  # instance | class | scramble | scramble_class
    corrupt_schedule: str = "block"  # block | mask_curriculum | blur_to_mask
    mask_ratio: float = 0.6
    patch_blur_ratio_min: float = 0.1
    patch_size: int = 4
    blur_sigma_min: float = 0.5
    blur_sigma_max: float = 3.0
    scramble_patch_size: int = 4
    vicreg_inv_weight: float = 0.0
    vicreg_var_weight: float = 0.0
    vicreg_cov_weight: float = 0.0

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


def resolve_training_mode(cfg: TrainConfig) -> TrainConfig:
    """Map high-level training modes to encoder/target/anchor settings."""
    if cfg.training_mode == "jepa_ema":
        return cfg
    if cfg.training_mode == "latent_triplet":
        # One encoder, latent-space JEPA + triplet regularizer (no EMA teacher).
        cfg.use_ema_target = False
        return cfg
    if cfg.training_mode == "latent_vicreg":
        # One encoder, JEPA invariance + VICReg anti-collapse (no EMA teacher).
        # anchor_mode from config: encoder_corrupt (direct) or predictor_corrupt (alignment head).
        cfg.use_ema_target = False
        return cfg
    raise ValueError(
        f"Unknown training_mode={cfg.training_mode!r}; "
        "expected jepa_ema, latent_triplet, or latent_vicreg"
    )


def train_one_epoch(
    model: TripletJEPA,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: TripletJEPALoss,
    device: torch.device,
    spec: DatasetSpec,
    cfg: TrainConfig,
    epoch: int,
) -> dict[str, float]:
    """One SSL epoch in latent space.

    latent_triplet mode (single encoder):
      anchor   = encoder(corrupt)           with grad
      positive = encoder(clean).detach()    stop-grad
      negative = encoder(scramble).detach() + different-class batch embedding

    latent_vicreg mode (single encoder):
      Full VICReg: λ·MSE(inv) + μ·L_var + ν·L_cov on encoder(corrupt/clean).
      Align mode: invariance on predictor(encoder(corrupt)) vs encoder(clean).
      JEPA cosine is logged but not optimized when VICReg weights are active.

    jepa_ema mode (default):
      anchor   = predictor(encoder(corrupt))
      positive = target_encoder(clean).detach()
    """
    model.train()
    totals: dict[str, float] = {"loss": 0.0, "jepa": 0.0}
    if cfg.triplet_weight > 0:
        totals["triplet"] = 0.0
    use_vicreg = (
        cfg.vicreg_inv_weight > 0
        or cfg.vicreg_var_weight > 0
        or cfg.vicreg_cov_weight > 0
    )
    if use_vicreg:
        totals["vicreg_inv"] = 0.0
        totals["vicreg_var"] = 0.0
        totals["vicreg_cov"] = 0.0
    n = 0
    progress = corrupt_progress(epoch, cfg.epochs)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Context view: corrupted image (mask and/or blur). Target view: clean augmentation.
        corrupt = make_corrupt_view(
            images,
            schedule=cfg.corrupt_schedule,
            progress=progress,
            mask_ratio=cfg.mask_ratio,
            patch_blur_ratio_min=cfg.patch_blur_ratio_min,
            blur_sigma_min=cfg.blur_sigma_min,
            blur_sigma_max=cfg.blur_sigma_max,
            min_block=cfg.patch_size,
        )

        z_anchor, z_positive = model(corrupt, images, anchor_mode=cfg.anchor_mode)
        z_pos = z_positive.detach()  # stop-grad on positive path

        z_neg = None
        z_neg_extra = None
        z_vicreg_a = None
        z_vicreg_b = None
        z_inv_a = None
        z_inv_b = None
        if cfg.triplet_weight > 0:
            if cfg.negative_mode == "scramble":
                neg_view = scramble_patches(images, patch_size=cfg.scramble_patch_size)
                z_neg = model.encode_target(neg_view).detach()
            elif cfg.negative_mode == "scramble_class":
                neg_view = scramble_patches(images, patch_size=cfg.scramble_patch_size)
                z_neg = model.encode_target(neg_view).detach()
                z_neg_extra = class_negatives(z_pos, labels)
            elif cfg.negative_mode == "class":
                z_neg = class_negatives(z_pos, labels)
            elif cfg.negative_mode == "instance":
                z_neg = instance_negatives(z_pos)
            else:
                raise ValueError(
                    f"Unknown negative_mode={cfg.negative_mode!r}; "
                    "expected instance, class, scramble, or scramble_class"
                )

        if use_vicreg:
            z_vicreg_a = model.encoder(corrupt)
            z_vicreg_b = model.encoder(images)
            if cfg.vicreg_inv_weight > 0:
                if cfg.anchor_mode == "predictor_corrupt":
                    z_inv_a = model.predictor(z_vicreg_a)
                    z_inv_b = z_vicreg_b.detach()
                else:
                    z_inv_a, z_inv_b = z_vicreg_a, z_vicreg_b

        loss, stats = criterion(
            z_anchor, z_pos, z_neg, z_neg_extra, z_vicreg_a, z_vicreg_b, z_inv_a, z_inv_b
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_target_encoder()

        bs = images.size(0)
        n += bs
        for k in totals:
            if k in stats:
                totals[k] += stats[k] * bs

    return {k: v / max(n, 1) for k, v in totals.items()}, progress


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def run_training(cfg: TrainConfig) -> dict:
    cfg = resolve_training_mode(cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    out = Path(cfg.output_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")

    print(
        f"training_mode={cfg.training_mode} | use_ema_target={cfg.use_ema_target} | "
        f"anchor_mode={cfg.anchor_mode} | negative_mode={cfg.negative_mode} | "
        f"corrupt_schedule={cfg.corrupt_schedule} | "
        f"triplet_weight={cfg.triplet_weight} | "
        f"vicreg_inv_weight={cfg.vicreg_inv_weight} | "
        f"vicreg_var_weight={cfg.vicreg_var_weight} | "
        f"vicreg_cov_weight={cfg.vicreg_cov_weight}"
    )

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
        use_ema_target=cfg.use_ema_target,
    ).to(device)
    params = model.param_count(cfg.anchor_mode)

    train_params = list(model.encoder.parameters())
    if cfg.anchor_mode == "predictor_corrupt":
        train_params += list(model.predictor.parameters())
    optimizer = optim.AdamW(
        train_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = TripletJEPALoss(
        margin=cfg.margin,
        triplet_weight=cfg.triplet_weight,
        vicreg_inv_weight=cfg.vicreg_inv_weight,
        vicreg_var_weight=cfg.vicreg_var_weight,
        vicreg_cov_weight=cfg.vicreg_cov_weight,
    )

    history: list[dict] = []
    t0 = time.time()
    use_vicreg = (
        cfg.vicreg_inv_weight > 0
        or cfg.vicreg_var_weight > 0
        or cfg.vicreg_cov_weight > 0
    )

    for epoch in range(1, cfg.epochs + 1):
        train_stats, progress = train_one_epoch(
            model, train_loader, optimizer, criterion, device, spec, cfg, epoch
        )
        scheduler.step()
        row: dict = {"epoch": epoch, **train_stats, "lr": scheduler.get_last_lr()[0]}
        if cfg.corrupt_schedule in {"blur_to_mask", "mask_curriculum"}:
            row["corrupt_progress"] = progress
            row["patch_blur_ratio"] = active_patch_blur_ratio(
                progress, cfg.patch_blur_ratio_min, cfg.mask_ratio
            )
            row["blur_sigma"] = cfg.blur_sigma_min + progress * (
                cfg.blur_sigma_max - cfg.blur_sigma_min
            )

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
        parts = [f"epoch {epoch:03d}/{cfg.epochs} | loss {row['loss']:.4f}"]
        if cfg.triplet_weight > 0:
            parts.append(f"jepa {row['jepa']:.4f}")
            parts.append(f"triplet {row['triplet']:.4f}")
        elif use_vicreg:
            if "vicreg_inv" in row:
                parts.append(f"inv {row['vicreg_inv']:.4f}")
            if "vicreg_var" in row:
                parts.append(f"var {row['vicreg_var']:.4f}")
            if "vicreg_cov" in row:
                parts.append(f"cov {row['vicreg_cov']:.4f}")
            parts.append(f"jepa(log) {row['jepa']:.4f}")
        else:
            parts.append(f"jepa {row['jepa']:.4f}")
            parts.append("triplet n/a")
        if cfg.corrupt_schedule in {"blur_to_mask", "mask_curriculum"}:
            parts.append(f"blur σ {row['blur_sigma']:.2f}")
            parts.append(f"patch blur {row['patch_blur_ratio']:.2f}")
        tag = " (" + ", ".join(parts[1:]) + ")"
        tag = parts[0] + tag
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
        "corrupt_schedule": cfg.corrupt_schedule,
        "patch_blur_ratio_min": cfg.patch_blur_ratio_min,
        "patch_size": cfg.patch_size,
        "blur_sigma_min": cfg.blur_sigma_min,
        "blur_sigma_max": cfg.blur_sigma_max,
        "training_mode": cfg.training_mode,
        "use_ema_target": cfg.use_ema_target,
        "anchor_mode": cfg.anchor_mode,
        "vicreg_inv_weight": cfg.vicreg_inv_weight,
        "vicreg_var_weight": cfg.vicreg_var_weight,
        "vicreg_cov_weight": cfg.vicreg_cov_weight,
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
