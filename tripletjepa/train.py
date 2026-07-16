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
    make_augmented_view,
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
    backbone: str = "resnet"  # resnet | vit
    vit_patch_size: int = 4
    vit_depth: int = 6
    vit_heads: int = 4
    vit_mlp_dim: int = 512
    ema_momentum: float = 0.996
    training_mode: str = "jepa_ema"  # jepa_ema | latent_triplet | latent_vicreg | latent_sigreg | latent_uniformity | latent_triplet_uniformity | latent_jepa_augment_uniformity
    use_ema_target: bool = True
    anchor_mode: str = "predictor_corrupt"  # predictor_corrupt | encoder_corrupt | encoder_clean

    # Loss
    margin: float = 0.2
    triplet_weight: float = 0.5
    negative_mode: str = "instance"  # instance | class | scramble | scramble_class
    corrupt_schedule: str = "block"  # block | block_curriculum | mask_curriculum | blur_to_mask
    mask_ratio: float = 0.6
    patch_blur_ratio_min: float = 0.1
    patch_size: int = 4
    blur_sigma_min: float = 0.5
    blur_sigma_max: float = 3.0
    scramble_patch_size: int = 4
    aug_brightness: float = 0.4
    aug_contrast: float = 0.4
    aug_saturation: float = 0.4
    aug_hue: float = 0.1
    aug_align_weight: float = 0.0
    align_weight: float = 1.0
    vicreg_inv_weight: float = 0.0
    vicreg_var_weight: float = 0.0
    vicreg_cov_weight: float = 0.0
    sigreg_weight: float = 0.0
    sigreg_num_slices: int = 256
    uniformity_weight: float = 0.0
    uniformity_t: float = 2.0
    proto_weight: float = 0.0
    proto_num: int = 100
    proto_temperature: float = 0.1
    sinkhorn_iters: int = 3
    sinkhorn_eps: float = 0.05

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
    resume: str | None = None
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
    if cfg.training_mode == "latent_sigreg":
        # One encoder, LeJEPA-style MSE invariance + SIGReg anti-collapse (no EMA teacher).
        cfg.use_ema_target = False
        return cfg
    if cfg.training_mode == "latent_uniformity":
        # One encoder, cosine JEPA alignment + hypersphere uniformity (Wang & Isola).
        cfg.use_ema_target = False
        return cfg
    if cfg.training_mode == "latent_triplet_uniformity":
        # Cosine JEPA align + cosine triplet (same-image scramble neg) + uniformity.
        cfg.use_ema_target = False
        cfg.negative_mode = "scramble"
        return cfg
    if cfg.training_mode == "latent_jepa_augment_uniformity":
        # JEPA on corrupt path + encoder(aug) align/triplet vs clean + uniformity on clean/aug.
        cfg.use_ema_target = False
        cfg.negative_mode = "scramble"
        if cfg.aug_align_weight <= 0:
            cfg.aug_align_weight = 1.0
        return cfg
    raise ValueError(
        f"Unknown training_mode={cfg.training_mode!r}; "
        "expected jepa_ema, latent_triplet, latent_vicreg, latent_sigreg, "
        "latent_uniformity, latent_triplet_uniformity, or latent_jepa_augment_uniformity"
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
    steps_per_epoch: int,
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

    latent_sigreg mode (single encoder):
      LeJEPA-style: (1-λ)·MSE(inv) + λ·SIGReg(encoder embeddings).
      Align mode: MSE(predictor(corrupt), encoder(clean)).
      SIGReg on concat(encoder(corrupt), encoder(clean)).

    latent_uniformity mode (single encoder):
      Cosine(predictor(corrupt), encoder(clean)) + w·Uniformity(encoder embeddings).
      Optional SwAV: unsupervised prototypes (no labels) swapped prediction on sphere.

    latent_triplet_uniformity mode (single encoder):
      Cosine align + λ·cosine_triplet(anchor, pos, scramble_neg) + w·Uniformity.
      Negative = encoder(scramble(same image)).detach() — label-free distortion.

    latent_jepa_augment_uniformity mode (single encoder):
      JEPA: cosine(predictor(corrupt), encoder(clean)).
      Aug align + triplet: encoder(aug) vs encoder(clean) vs scramble(aug) — no predictor.
      Uniformity on encoder(clean) batch only (not aug, avoids fighting aug_align).

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
    use_sigreg = cfg.sigreg_weight > 0
    use_uniformity = cfg.uniformity_weight > 0
    use_proto = cfg.proto_weight > 0
    use_triplet_uniformity = cfg.training_mode == "latent_triplet_uniformity"
    use_jepa_augment = cfg.training_mode == "latent_jepa_augment_uniformity"
    use_sphere = use_uniformity or use_proto or use_triplet_uniformity
    if use_vicreg:
        totals["vicreg_inv"] = 0.0
        totals["vicreg_var"] = 0.0
        totals["vicreg_cov"] = 0.0
    if use_sigreg:
        totals["sigreg_inv"] = 0.0
        totals["sigreg"] = 0.0
    if use_uniformity or use_triplet_uniformity or use_jepa_augment:
        totals["align"] = 0.0
    if use_uniformity or use_jepa_augment:
        totals["uniformity"] = 0.0
    elif use_proto:
        totals["align"] = 0.0
    if (use_triplet_uniformity or use_jepa_augment) and cfg.triplet_weight > 0:
        totals["triplet"] = 0.0
    if use_jepa_augment:
        totals["aug_align"] = 0.0
    if use_proto:
        totals["swav"] = 0.0
    n = 0
    progress = corrupt_progress(epoch, cfg.epochs)

    for batch_idx, (images, labels) in enumerate(loader):
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

        z_neg = None
        z_neg_extra = None
        z_vicreg_a = None
        z_vicreg_b = None
        z_inv_a = None
        z_inv_b = None
        z_sigreg = None
        z_uniformity = None
        z_enc_aug = None
        prototypes = None
        z_proto_a = None
        z_proto_b = None
        global_step = (epoch - 1) * steps_per_epoch + batch_idx

        if use_jepa_augment:
            augmented = make_augmented_view(
                images,
                brightness=cfg.aug_brightness,
                contrast=cfg.aug_contrast,
                saturation=cfg.aug_saturation,
                hue=cfg.aug_hue,
            )
            z_clean_enc = model.encoder(images)
            z_aug_enc = model.encoder(augmented)
            z_corrupt_enc = model.encoder(corrupt)
            z_inv_a = model.predictor(z_corrupt_enc)
            z_inv_b = z_clean_enc.detach()
            z_enc_aug = z_aug_enc
            z_anchor, z_pos = z_inv_a, z_inv_b
            # Uniformity on clean only — concat(clean, aug) fights aug_align by spreading pairs apart.
            z_uniformity = z_clean_enc

            if cfg.triplet_weight > 0:
                z_ref = z_clean_enc.detach()
                if cfg.negative_mode == "scramble":
                    neg_view = scramble_patches(augmented, patch_size=cfg.scramble_patch_size)
                    z_neg = model.encode_target(neg_view).detach()
                elif cfg.negative_mode == "scramble_class":
                    neg_view = scramble_patches(images, patch_size=cfg.scramble_patch_size)
                    z_neg = model.encode_target(neg_view).detach()
                    z_neg_extra = class_negatives(z_ref, labels)
                elif cfg.negative_mode == "class":
                    z_neg = class_negatives(z_ref, labels)
                elif cfg.negative_mode == "instance":
                    z_neg = instance_negatives(z_ref)
                else:
                    raise ValueError(
                        f"Unknown negative_mode={cfg.negative_mode!r}; "
                        "expected instance, class, scramble, or scramble_class"
                    )
        else:
            z_anchor, z_positive = model(corrupt, images, anchor_mode=cfg.anchor_mode)
            z_pos = z_positive.detach()  # stop-grad on positive path

            if cfg.triplet_weight > 0 or use_triplet_uniformity:
                if cfg.negative_mode == "scramble" or use_triplet_uniformity:
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

            if use_vicreg or use_sigreg or use_sphere:
                z_vicreg_a = model.encoder(corrupt)
                z_vicreg_b = model.encoder(images)
                if cfg.vicreg_inv_weight > 0 or use_sigreg or use_sphere:
                    if cfg.anchor_mode == "predictor_corrupt":
                        z_inv_a = model.predictor(z_vicreg_a)
                        z_inv_b = z_vicreg_b.detach()
                    else:
                        z_inv_a, z_inv_b = z_vicreg_a, z_vicreg_b
                if use_sigreg:
                    z_sigreg = torch.cat([z_vicreg_a, z_vicreg_b], dim=0)
                if use_uniformity:
                    z_uniformity = torch.cat([z_vicreg_a, z_vicreg_b], dim=0)
                if use_proto:
                    if model.prototype_bank is None:
                        raise ValueError("proto_weight > 0 requires model.prototype_bank")
                    prototypes = model.prototype_bank()
                    # SwAV views: predicted corrupt path vs clean encoder (grad on both).
                    z_proto_a = z_inv_a if cfg.anchor_mode == "predictor_corrupt" else z_vicreg_a
                    z_proto_b = z_vicreg_b

        loss, stats = criterion(
            z_anchor,
            z_pos,
            z_neg,
            z_neg_extra,
            z_vicreg_a,
            z_vicreg_b,
            z_inv_a,
            z_inv_b,
            z_sigreg,
            z_uniformity,
            z_enc_aug,
            prototypes,
            z_proto_a,
            z_proto_b,
            global_step,
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


def checkpoint_last_path(out: Path) -> Path:
    return out / "checkpoint_last.pt"


def checkpoint_best_path(out: Path) -> Path:
    return out / "checkpoint_best.pt"


def best_knn_from_history(history: list[dict]) -> tuple[float, int]:
    best_knn = -1.0
    best_epoch = 0
    for row in history:
        knn = row.get("knn")
        if knn is None:
            continue
        if knn > best_knn:
            best_knn = float(knn)
            best_epoch = int(row["epoch"])
    return best_knn, best_epoch


def _checkpoint_config_keys() -> tuple[str, ...]:
    return (
        "training_mode",
        "embed_dim",
        "backbone",
        "vit_patch_size",
        "vit_depth",
        "anchor_mode",
        "use_ema_target",
        "proto_num",
        "proto_weight",
        "epochs",
        "run_name",
        "dataset",
    )


def validate_resume_config(cfg: TrainConfig, saved_cfg: dict) -> None:
    mismatches = []
    for key in _checkpoint_config_keys():
        if saved_cfg.get(key) != getattr(cfg, key):
            mismatches.append(f"{key}: checkpoint={saved_cfg.get(key)!r} config={getattr(cfg, key)!r}")
    if mismatches:
        raise ValueError(
            "Resume config mismatch (use the same config as the original run):\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )


def save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    cfg: TrainConfig,
    model: TripletJEPA,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    history: list[dict],
    elapsed_sec: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "config": asdict(cfg),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "elapsed_sec": elapsed_sec,
        },
        path,
    )


def load_training_checkpoint(
    path: str | Path,
    device: torch.device,
) -> dict:
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def run_training(cfg: TrainConfig) -> dict:
    cfg = resolve_training_mode(cfg)
    device = resolve_device(cfg.device)

    out = Path(cfg.output_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)

    resume_path = Path(cfg.resume) if cfg.resume else None
    ckpt: dict | None = None
    start_epoch = 1
    history: list[dict] = []
    elapsed_before = 0.0

    if resume_path is not None:
        ckpt = load_training_checkpoint(resume_path, device)
        validate_resume_config(cfg, ckpt["config"])
        start_epoch = int(ckpt["epoch"]) + 1
        history = list(ckpt.get("history", []))
        elapsed_before = float(ckpt.get("elapsed_sec", 0.0))
        if start_epoch > cfg.epochs:
            raise ValueError(
                f"Checkpoint epoch {ckpt['epoch']} >= target epochs {cfg.epochs}; nothing to resume"
            )
        print(f"Resuming from {resume_path} at epoch {start_epoch}/{cfg.epochs}")
    else:
        set_seed(cfg.seed)

    cfg.to_json(out / "config.json")

    print(
        f"training_mode={cfg.training_mode} | backbone={cfg.backbone} | "
        f"use_ema_target={cfg.use_ema_target} | "
        f"anchor_mode={cfg.anchor_mode} | negative_mode={cfg.negative_mode} | "
        f"corrupt_schedule={cfg.corrupt_schedule} | "
        f"triplet_weight={cfg.triplet_weight} | "
        f"align_weight={cfg.align_weight} | "
        f"aug_align_weight={cfg.aug_align_weight} | "
        f"margin={cfg.margin} | "
        f"vicreg_inv_weight={cfg.vicreg_inv_weight} | "
        f"vicreg_var_weight={cfg.vicreg_var_weight} | "
        f"vicreg_cov_weight={cfg.vicreg_cov_weight} | "
        f"sigreg_weight={cfg.sigreg_weight} | "
        f"sigreg_num_slices={cfg.sigreg_num_slices} | "
        f"uniformity_weight={cfg.uniformity_weight} | "
        f"uniformity_t={cfg.uniformity_t} | "
        f"proto_weight={cfg.proto_weight} | "
        f"proto_num={cfg.proto_num}"
    )

    if cfg.backbone.lower() in {"vit", "small_vit"}:
        print(
            f"vit_patch_size={cfg.vit_patch_size} | vit_depth={cfg.vit_depth} | "
            f"vit_heads={cfg.vit_heads} | vit_mlp_dim={cfg.vit_mlp_dim} | "
            f"embed_dim={cfg.embed_dim}"
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
        num_prototypes=cfg.proto_num if cfg.proto_weight > 0 else 0,
        backbone=cfg.backbone,
        image_size=spec.image_size,
        vit_patch_size=cfg.vit_patch_size,
        vit_depth=cfg.vit_depth,
        vit_heads=cfg.vit_heads,
        vit_mlp_dim=cfg.vit_mlp_dim,
    ).to(device)
    params = model.param_count(cfg.anchor_mode)
    print(f"params encoder={params['encoder']} predictor={params['predictor']} total={params['total_trainable']}")

    train_params = list(model.encoder.parameters())
    if cfg.anchor_mode == "predictor_corrupt":
        train_params += list(model.predictor.parameters())
    if model.prototype_bank is not None:
        train_params += list(model.prototype_bank.parameters())
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
        sigreg_weight=cfg.sigreg_weight,
        sigreg_num_slices=cfg.sigreg_num_slices,
        uniformity_weight=cfg.uniformity_weight,
        uniformity_t=cfg.uniformity_t,
        proto_weight=cfg.proto_weight,
        proto_num=cfg.proto_num,
        proto_temperature=cfg.proto_temperature,
        sinkhorn_iters=cfg.sinkhorn_iters,
        sinkhorn_eps=cfg.sinkhorn_eps,
        triplet_cosine=(cfg.training_mode == "latent_triplet_uniformity"),
        jepa_augment_triplet_uniformity=(
            cfg.training_mode == "latent_jepa_augment_uniformity"
        ),
        aug_align_weight=cfg.aug_align_weight,
        align_weight=cfg.align_weight,
    )

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    best_knn, best_epoch = best_knn_from_history(history)

    t0 = time.time()
    use_vicreg = (
        cfg.vicreg_inv_weight > 0
        or cfg.vicreg_var_weight > 0
        or cfg.vicreg_cov_weight > 0
    )
    use_sigreg = cfg.sigreg_weight > 0
    use_uniformity = cfg.uniformity_weight > 0
    use_proto = cfg.proto_weight > 0
    use_triplet_uniformity = cfg.training_mode == "latent_triplet_uniformity"
    use_jepa_augment = cfg.training_mode == "latent_jepa_augment_uniformity"
    steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_stats, progress = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            spec,
            cfg,
            epoch,
            steps_per_epoch,
        )
        scheduler.step()
        row: dict = {"epoch": epoch, **train_stats, "lr": scheduler.get_last_lr()[0]}
        if cfg.corrupt_schedule in {"blur_to_mask", "mask_curriculum", "block_curriculum"}:
            row["corrupt_progress"] = progress
            row["active_mask_ratio"] = active_patch_blur_ratio(
                progress, cfg.patch_blur_ratio_min, cfg.mask_ratio
            )
        if cfg.corrupt_schedule in {"blur_to_mask", "mask_curriculum"}:
            row["patch_blur_ratio"] = row["active_mask_ratio"]
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
        if cfg.triplet_weight > 0 and not use_triplet_uniformity and not use_jepa_augment:
            parts.append(f"jepa {row['jepa']:.4f}")
            parts.append(f"triplet {row['triplet']:.4f}")
        elif use_triplet_uniformity or use_jepa_augment or use_uniformity or use_proto:
            if "align" in row:
                parts.append(f"align {row['align']:.4f}")
            if "aug_align" in row:
                parts.append(f"aug_align {row['aug_align']:.4f}")
            if "uniformity" in row:
                parts.append(f"unif {row['uniformity']:.4f}")
            if "triplet" in row:
                parts.append(f"triplet {row['triplet']:.4f}")
            if "swav" in row:
                parts.append(f"swav {row['swav']:.4f}")
            parts.append(f"jepa(log) {row['jepa']:.4f}")
        elif use_sigreg:
            if "sigreg_inv" in row:
                parts.append(f"inv {row['sigreg_inv']:.4f}")
            if "sigreg" in row:
                parts.append(f"sigreg {row['sigreg']:.4f}")
            parts.append(f"jepa(log) {row['jepa']:.4f}")
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
        if cfg.corrupt_schedule == "block_curriculum":
            parts.append(f"mask {row['active_mask_ratio']:.2f}")
        if cfg.corrupt_schedule in {"blur_to_mask", "mask_curriculum"}:
            parts.append(f"blur σ {row['blur_sigma']:.2f}")
            parts.append(f"patch blur {row['patch_blur_ratio']:.2f}")
        tag = " (" + ", ".join(parts[1:]) + ")"
        tag = parts[0] + tag
        if "knn" in row:
            tag += f" | k-NN {row['knn']:.3f} | linear {row['linear_probe']:.3f}"
        print(tag)

        elapsed_now = elapsed_before + (time.time() - t0)
        save_training_checkpoint(
            checkpoint_last_path(out),
            epoch=epoch,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            elapsed_sec=elapsed_now,
        )

        if "knn" in row and row["knn"] > best_knn:
            best_knn = float(row["knn"])
            best_epoch = epoch
            save_training_checkpoint(
                checkpoint_best_path(out),
                epoch=epoch,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                elapsed_sec=elapsed_now,
            )
            print(f"  -> new best k-NN {best_knn:.3f} @ epoch {best_epoch} (saved checkpoint_best.pt)")

    elapsed = elapsed_before + (time.time() - t0)

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
        "sigreg_weight": cfg.sigreg_weight,
        "sigreg_num_slices": cfg.sigreg_num_slices,
        "uniformity_weight": cfg.uniformity_weight,
        "uniformity_t": cfg.uniformity_t,
        "proto_weight": cfg.proto_weight,
        "proto_num": cfg.proto_num,
        "proto_temperature": cfg.proto_temperature,
        "margin": cfg.margin,
        "epochs": cfg.epochs,
        "params": params,
        "elapsed_sec": round(elapsed, 1),
        "device": str(device),
        "best_knn": best_knn if best_epoch > 0 else None,
        "best_epoch": best_epoch if best_epoch > 0 else None,
        "final": history[-1],
        "history": history,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    return summary
