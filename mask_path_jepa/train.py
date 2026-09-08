from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mask_path_jepa.data import DatasetSpec, get_dataloaders
from mask_path_jepa.eval import evaluate_encoder
from mask_path_jepa.losses import (
    JEPALoss,
    active_nn_tau,
    gated_centroid_nn_loss,
    intra_centroid_pull_loss,
    pull_push_centroid_loss,
    softmax_centroid_nn_loss,
    uniformity_loss,
)
from mask_path_jepa.models import JEPA
from mask_path_jepa.teacher import (
    TeacherFeatureBank,
    teacher_weight_at_epoch,
    teacher_weighted_align_loss,
)
from mask_path_jepa.views import (
    active_patch_blur_ratio,
    class_negatives,
    corrupt_progress,
    distortion_ladder,
    instance_negatives,
    make_augmented_view,
    make_centroid_aug_stack,
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
    backbone: str = "resnet"  # resnet | resnet18 | resnet50 | vit | vit_tiny
    vit_patch_size: int = 4
    vit_depth: int = 6
    vit_heads: int = 4
    vit_mlp_dim: int = 512
    ema_momentum: float = 0.996
    training_mode: str = "jepa_ema"  # jepa_ema | jepa_gated_centroid_nn | latent_triplet | ...
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
    jepa_infonce_weight: float = 1.0
    aug_infonce_weight: float = 1.0
    infonce_temperature: float = 0.1
    # Gated / softmax centroid-NN (in-batch neighbors on aug means)
    nn_mode: str = "pull_push"  # pull_push | softmax | gated
    nn_temperature: float = 0.05
    nn_num_augs: int = 6
    nn_tau: float = 0.9995
    nn_tau_end: float = 0.9985
    nn_tau_anneal_epochs: int = 10
    nn_k: int = 1
    nn_weight: float = 1.0
    nn_push_weight: float = 1.0
    nn_intra_weight: float = 1.0
    nn_mutual: bool = False
    nn_soft_weight: bool = True
    nn_sim_gap: float = 0.0005
    nn_encode_chunk: int = 256
    nn_precompute_augs: bool = True
    nn_aug_cache_dir: str = "./data/aug_banks"
    nn_aug_force_recompute: bool = False
    rank_weight: float = 1.0
    rank_similarity: float = 0.9
    rank_transparency: float = 0.5
    teacher_features_path: str | None = None
    teacher_align_weight: float = 0.0
    teacher_align_temperature: float = 0.1
    teacher_anneal_start_epoch: int = 1
    teacher_anneal_end_epoch: int = 70
    vicreg_inv_weight: float = 0.0
    vicreg_var_weight: float = 0.0
    vicreg_cov_weight: float = 0.0
    sigreg_weight: float = 0.0
    sigreg_inv_weight: float = 0.0
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
    run_name: str = "mask_path_jepa"
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
    if cfg.training_mode == "jepa_gated_centroid_nn":
        # Encoder-only centroid losses. Default: hard pull+push only (no JEPA / unif / InfoNCE).
        cfg.use_ema_target = False
        cfg.align_weight = 0.0
        cfg.jepa_infonce_weight = 0.0
        cfg.triplet_weight = 0.0
        if cfg.nn_weight <= 0:
            cfg.nn_weight = 1.0
        if not cfg.nn_mode:
            cfg.nn_mode = "pull_push"
        # Do not force uniformity — user wants attributable single-objective runs.
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
    if cfg.training_mode == "latent_infonce_jepa_augment":
        # InfoNCE on predictor(mask) vs clean batch + optional encoder(aug) vs clean batch.
        cfg.use_ema_target = False
        return cfg
    if cfg.training_mode in {"latent_infonce_jepa_vicreg", "latent_infonce_jepa_mse_var_cov"}:
        # Mask JEPA (cosine and/or InfoNCE via weights) + VICReg/MSE-var-cov on aug/clean.
        cfg.use_ema_target = False
        cfg.aug_infonce_weight = 0.0
        return cfg
    if cfg.training_mode == "latent_infonce_jepa_sigreg":
        # Mask JEPA (cosine and/or InfoNCE via weights) + SIGReg on aug/clean.
        cfg.use_ema_target = False
        cfg.aug_infonce_weight = 0.0
        return cfg
    if cfg.training_mode == "latent_distortion_ranking":
        # Encoder-only: monotonic cosine distance from clean across corruption ladder.
        cfg.use_ema_target = False
        cfg.anchor_mode = "encoder_clean"
        return cfg
    raise ValueError(
        f"Unknown training_mode={cfg.training_mode!r}; "
        "expected jepa_ema, jepa_gated_centroid_nn, latent_triplet, latent_vicreg, latent_sigreg, "
        "latent_uniformity, latent_triplet_uniformity, latent_jepa_augment_uniformity, "
        "latent_infonce_jepa_augment, latent_infonce_jepa_vicreg, latent_infonce_jepa_mse_var_cov, "
        "latent_infonce_jepa_sigreg, or latent_distortion_ranking"
    )


def train_one_epoch(
    model: JEPA,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: JEPALoss,
    device: torch.device,
    spec: DatasetSpec,
    cfg: TrainConfig,
    epoch: int,
    steps_per_epoch: int,
    teacher_bank: TeacherFeatureBank | None = None,
    teacher_weight: float = 0.0,
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

    latent_infonce_jepa_augment mode (single encoder):
      L_jepa = InfoNCE(predictor(corrupt), encoder(clean).detach(), batch negatives).
      L_aug  = InfoNCE(encoder(aug), encoder(clean).detach(), batch negatives).
      Cosine align / aug_align logged as diagnostics; optional via align_weight / aug_align_weight.

    latent_infonce_jepa_vicreg / latent_infonce_jepa_mse_var_cov (single encoder):
      L_jepa = InfoNCE(predictor(corrupt), encoder(clean).detach(), batch negatives).
      L_2nd  = λ·MSE(encoder(aug), encoder(clean)) + μ·L_var + ν·L_cov on both aug/clean batches.
      Replaces aug InfoNCE with information-maximization regularizer (VICReg-style spread).

    latent_infonce_jepa_sigreg (single encoder):
      L_jepa = InfoNCE(predictor(corrupt), encoder(clean).detach(), batch negatives).
      L_2nd  = λ·MSE(encoder(aug), encoder(clean)) + μ·SIGReg(concat(aug, clean)).
      LeJEPA-style Gaussian regularizer instead of explicit var/cov.

    latent_distortion_ranking mode (encoder + optional predictor):
      Ladder: clean -> 0.9 similarity -> 0.5 transparency -> hard mask (shared patches).
      L_rank on encoder distances; optional InfoNCE(predictor(hard), clean) + InfoNCE(encoder(aug), clean).

    jepa_ema mode (default):
      anchor   = predictor(encoder(corrupt))
      positive = target_encoder(clean).detach()

    jepa_gated_centroid_nn mode:
      Encoder-only. Soft/gated neighbor loss on per-image aug centroids
      (M augs → mean embedding) + optional uniformity. No JEPA / corrupt / EMA.
    """
    model.train()
    use_distortion_ranking = cfg.training_mode == "latent_distortion_ranking"
    use_gated_centroid_nn = cfg.training_mode == "jepa_gated_centroid_nn"
    totals: dict[str, float] = {"loss": 0.0}
    if not use_distortion_ranking and not use_gated_centroid_nn:
        totals["jepa"] = 0.0
    if cfg.triplet_weight > 0:
        totals["triplet"] = 0.0
    use_vicreg = (
        cfg.vicreg_inv_weight > 0
        or cfg.vicreg_var_weight > 0
        or cfg.vicreg_cov_weight > 0
    )
    use_sigreg = cfg.sigreg_weight > 0
    # Centroid-NN applies uniformity on aug-centroids; skip criterion path to avoid double hit.
    use_uniformity = cfg.uniformity_weight > 0 and not use_gated_centroid_nn
    use_centroid_uniformity = use_gated_centroid_nn and cfg.uniformity_weight > 0
    use_proto = cfg.proto_weight > 0
    use_triplet_uniformity = cfg.training_mode == "latent_triplet_uniformity"
    use_jepa_augment = cfg.training_mode == "latent_jepa_augment_uniformity"
    use_infonce_jepa_augment = cfg.training_mode == "latent_infonce_jepa_augment"
    use_infonce_jepa_vicreg = cfg.training_mode in {
        "latent_infonce_jepa_vicreg",
        "latent_infonce_jepa_mse_var_cov",
    }
    use_infonce_jepa_sigreg = cfg.training_mode == "latent_infonce_jepa_sigreg"
    use_dual_view = (
        use_jepa_augment
        or use_infonce_jepa_augment
        or use_infonce_jepa_vicreg
        or use_infonce_jepa_sigreg
    )
    use_sphere = use_uniformity or use_proto or use_triplet_uniformity
    if use_vicreg:
        totals["vicreg_inv"] = 0.0
        totals["vicreg_var"] = 0.0
        totals["vicreg_cov"] = 0.0
    if use_sigreg:
        totals["sigreg_inv"] = 0.0
        totals["sigreg"] = 0.0
    if use_uniformity or use_triplet_uniformity or use_dual_view:
        totals["align"] = 0.0
    if use_uniformity or use_jepa_augment or use_infonce_jepa_augment or use_infonce_jepa_vicreg or use_infonce_jepa_sigreg:
        totals["uniformity"] = 0.0
    elif use_proto:
        totals["align"] = 0.0
    if (use_triplet_uniformity or use_jepa_augment) and cfg.triplet_weight > 0:
        totals["triplet"] = 0.0
    if use_jepa_augment:
        totals["aug_align"] = 0.0
    if use_infonce_jepa_augment:
        totals["jepa_infonce"] = 0.0
        totals["aug_infonce"] = 0.0
        totals["aug_align"] = 0.0
    if use_infonce_jepa_vicreg:
        totals["jepa_infonce"] = 0.0
        totals["vicreg_inv"] = 0.0
        totals["vicreg_var"] = 0.0
        totals["vicreg_cov"] = 0.0
        totals["aug_align"] = 0.0
        totals["align"] = 0.0
    if use_infonce_jepa_sigreg:
        totals["jepa_infonce"] = 0.0
        totals["sigreg_inv"] = 0.0
        totals["sigreg"] = 0.0
        totals["aug_align"] = 0.0
        totals["align"] = 0.0
    if use_distortion_ranking:
        totals["rank"] = 0.0
        totals["rank_violation"] = 0.0
        for i in range(1, 4):
            totals[f"rank_d{i}"] = 0.0
        if cfg.jepa_infonce_weight > 0:
            totals["jepa_infonce"] = 0.0
        if cfg.aug_infonce_weight > 0:
            totals["aug_infonce"] = 0.0
        if cfg.align_weight > 0 or cfg.jepa_infonce_weight > 0:
            totals["align"] = 0.0
        if cfg.aug_align_weight > 0 or cfg.aug_infonce_weight > 0:
            totals["aug_align"] = 0.0
    if use_proto:
        totals["swav"] = 0.0
    if use_gated_centroid_nn and cfg.nn_weight > 0:
        totals["nn"] = 0.0
        totals["nn_weighted"] = 0.0
        totals["nn_pairs"] = 0.0
        totals["nn_mean_sim"] = 0.0
        totals["nn_frac_gated"] = 0.0
        totals["nn_tau"] = 0.0
        totals["nn_active_steps"] = 0.0
        totals["nn_total_pairs"] = 0.0
        totals["nn_active_frac"] = 0.0
        if cfg.nn_mode == "softmax":
            totals["nn_entropy"] = 0.0
            totals["nn_max_p"] = 0.0
            totals["nn_temp"] = 0.0
        if cfg.nn_mode == "pull_push":
            totals["nn_pull"] = 0.0
            totals["nn_push"] = 0.0
            totals["nn_near_sim"] = 0.0
            totals["nn_far_sim"] = 0.0
        if cfg.nn_intra_weight > 0:
            totals["intra"] = 0.0
            totals["intra_mean_sim"] = 0.0
        if cfg.uniformity_weight > 0:
            totals["uniformity"] = 0.0
    if teacher_bank is not None and teacher_weight > 0:
        totals["teacher_align"] = 0.0
    n = 0
    n_steps = 0
    nn_active_steps = 0
    nn_total_pairs = 0
    progress = corrupt_progress(epoch, cfg.epochs)
    nn_tau = active_nn_tau(
        epoch,
        cfg.epochs,
        cfg.nn_tau,
        cfg.nn_tau_end,
        anneal_epochs=cfg.nn_tau_anneal_epochs,
    )

    for batch_idx, batch in enumerate(loader):
        precomputed_augs = None
        if teacher_bank is not None:
            images, labels, sample_idx = batch
            sample_idx = sample_idx.to(device, non_blocking=True)
        elif (
            isinstance(batch, (tuple, list))
            and len(batch) == 3
            and torch.is_tensor(batch[2])
            and batch[2].dim() == 5
        ):
            images, labels, precomputed_augs = batch
            sample_idx = None
            precomputed_augs = precomputed_augs.to(device, non_blocking=True)
        else:
            images, labels = batch
            sample_idx = None

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        corrupt = None
        if not use_distortion_ranking and not use_gated_centroid_nn:
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
        z_clean_for_teacher = None
        z_rank_levels = None
        global_step = (epoch - 1) * steps_per_epoch + batch_idx

        if use_gated_centroid_nn:
            # No JEPA forward — loss comes only from centroid NN (+ uniformity) below.
            loss = torch.zeros((), device=device)
            stats: dict[str, float] = {}
        elif use_distortion_ranking:
            ladder = distortion_ladder(
                images,
                mask_ratio=cfg.mask_ratio,
                patch_size=cfg.patch_size,
                similarity=cfg.rank_similarity,
                transparency=cfg.rank_transparency,
            )
            b, k, c, h, w = ladder.shape
            z_ladder = model.encoder(ladder.view(b * k, c, h, w)).view(b, k, -1)
            z_anchor = z_ladder[:, 0]
            z_rank_levels = z_ladder[:, 1:]
            z_pos = z_anchor.detach()
            z_inv_a = None
            z_inv_b = None
            z_enc_aug = None
            use_rank_jepa = cfg.jepa_infonce_weight > 0 or cfg.align_weight > 0
            use_rank_aug = cfg.aug_infonce_weight > 0 or cfg.aug_align_weight > 0
            if use_rank_jepa:
                z_inv_a = model.predictor(z_ladder[:, 3])
                z_inv_b = z_anchor.detach()
            if use_rank_aug:
                augmented = make_augmented_view(
                    images,
                    brightness=cfg.aug_brightness,
                    contrast=cfg.aug_contrast,
                    saturation=cfg.aug_saturation,
                    hue=cfg.aug_hue,
                )
                z_enc_aug = model.encoder(augmented)
                if z_inv_b is None:
                    z_inv_b = z_anchor.detach()
        elif use_dual_view:
            augmented = make_augmented_view(
                images,
                brightness=cfg.aug_brightness,
                contrast=cfg.aug_contrast,
                saturation=cfg.aug_saturation,
                hue=cfg.aug_hue,
            )
            z_clean_enc = model.encoder(images)
            z_clean_for_teacher = z_clean_enc
            z_aug_enc = model.encoder(augmented)
            z_corrupt_enc = model.encoder(corrupt)
            z_inv_a = model.predictor(z_corrupt_enc)
            z_inv_b = z_clean_enc.detach()
            z_enc_aug = z_aug_enc
            z_anchor, z_pos = z_inv_a, z_inv_b
            if use_infonce_jepa_vicreg:
                z_vicreg_a = z_aug_enc
                z_vicreg_b = z_clean_enc
            elif use_infonce_jepa_sigreg:
                z_vicreg_a = z_aug_enc
                z_vicreg_b = z_clean_enc
                z_sigreg = torch.cat([z_aug_enc, z_clean_enc], dim=0)
            # Uniformity on clean only — concat(clean, aug) fights aug_align by spreading pairs apart.
            z_uniformity = z_clean_enc

            if use_jepa_augment and cfg.triplet_weight > 0:
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

        if not use_gated_centroid_nn:
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
                z_rank_levels,
                global_step,
            )
        if use_gated_centroid_nn and cfg.nn_weight > 0:
            if precomputed_augs is not None:
                stack = precomputed_augs
            else:
                stack = make_centroid_aug_stack(
                    images,
                    cfg.nn_num_augs,
                    brightness=cfg.aug_brightness,
                    contrast=cfg.aug_contrast,
                    saturation=cfg.aug_saturation,
                    hue=cfg.aug_hue,
                )
            bsz, n_aug, ch, hh, ww = stack.shape
            flat = stack.reshape(bsz * n_aug, ch, hh, ww)
            zs = []
            chunk = max(int(cfg.nn_encode_chunk), 1)
            for s in range(0, flat.size(0), chunk):
                zs.append(model.encoder(flat[s : s + chunk]))
            z_all = torch.cat(zs, dim=0).view(bsz, n_aug, -1)
            centroids = torch.nn.functional.normalize(z_all.mean(dim=1), dim=-1)
            if cfg.nn_mode == "pull_push":
                l_nn, nn_stats = pull_push_centroid_loss(
                    centroids,
                    push_weight=cfg.nn_push_weight,
                    mutual_pull=cfg.nn_mutual,
                )
                stats["nn_pull"] = nn_stats.get("nn_pull", 0.0)
                stats["nn_push"] = nn_stats.get("nn_push", 0.0)
                stats["nn_near_sim"] = nn_stats.get("nn_near_sim", 0.0)
                stats["nn_far_sim"] = nn_stats.get("nn_far_sim", 0.0)
                n_pairs = int(nn_stats["nn_pairs"])
                nn_total_pairs += n_pairs
                nn_active_steps += 1
            elif cfg.nn_mode == "softmax":
                l_nn, nn_stats = softmax_centroid_nn_loss(
                    centroids,
                    temperature=cfg.nn_temperature,
                    mutual=cfg.nn_mutual,
                )
                stats["nn_entropy"] = nn_stats.get("nn_entropy", 0.0)
                stats["nn_max_p"] = nn_stats.get("nn_max_p", 0.0)
                stats["nn_temp"] = float(cfg.nn_temperature)
                # Softmax always active
                n_pairs = int(bsz)
                nn_total_pairs += n_pairs
                nn_active_steps += 1
            else:
                l_nn, nn_stats = gated_centroid_nn_loss(
                    centroids,
                    tau=nn_tau,
                    k=cfg.nn_k,
                    mutual=cfg.nn_mutual,
                    soft_weight=cfg.nn_soft_weight,
                    sim_gap=cfg.nn_sim_gap,
                )
                n_pairs = int(nn_stats["nn_pairs"])
                nn_total_pairs += n_pairs
                if n_pairs > 0:
                    nn_active_steps += 1
                stats["nn_tau"] = float(nn_tau)
            loss = loss + cfg.nn_weight * l_nn
            stats["nn"] = l_nn.item()
            stats["nn_weighted"] = float(cfg.nn_weight) * l_nn.item()
            stats["nn_pairs"] = nn_stats["nn_pairs"]
            stats["nn_mean_sim"] = nn_stats["nn_mean_sim"]
            stats["nn_frac_gated"] = nn_stats["nn_frac_gated"]
            # Same-image: pull the M augs of each centroid together.
            if cfg.nn_intra_weight > 0:
                l_intra, intra_stats = intra_centroid_pull_loss(z_all)
                loss = loss + cfg.nn_intra_weight * l_intra
                stats["intra"] = l_intra.item()
                stats["intra_mean_sim"] = intra_stats.get("intra_mean_sim", 0.0)
            # Optional anti-collapse (off by default for clean pull/push ablations).
            if use_centroid_uniformity:
                l_unif = uniformity_loss(centroids, cfg.uniformity_t)
                loss = loss + cfg.uniformity_weight * l_unif
                stats["uniformity"] = l_unif.item()
        if teacher_bank is not None and teacher_weight > 0 and sample_idx is not None:
            if z_clean_for_teacher is None:
                z_clean_for_teacher = model.encoder(images)
            phi = teacher_bank.lookup(sample_idx)
            l_teacher = teacher_weighted_align_loss(
                z_clean_for_teacher,
                phi,
                cfg.teacher_align_temperature,
            )
            loss = loss + teacher_weight * l_teacher
            stats["teacher_align"] = l_teacher.item()
        stats["loss"] = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_target_encoder()

        bs = images.size(0)
        n += bs
        n_steps += 1
        for k in totals:
            if k in stats and k not in {"nn_active_steps", "nn_total_pairs", "nn_active_frac"}:
                totals[k] += stats[k] * bs

    out = {k: v / max(n, 1) for k, v in totals.items()}
    if use_gated_centroid_nn and cfg.nn_weight > 0:
        out["nn_active_steps"] = float(nn_active_steps)
        out["nn_total_pairs"] = float(nn_total_pairs)
        out["nn_active_frac"] = float(nn_active_steps) / float(max(n_steps, 1))
        out["nn_steps"] = float(n_steps)
    return out, progress


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
        if key == "epochs":
            saved_epochs = saved_cfg.get("epochs")
            if saved_epochs is not None and cfg.epochs < saved_epochs:
                mismatches.append(
                    f"epochs: checkpoint={saved_epochs!r} config={cfg.epochs!r} "
                    "(target must be >= checkpoint training budget)"
                )
            continue
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
    model: JEPA,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    history: list[dict],
    elapsed_sec: float,
    also_export_encoder: bool = False,
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
    # Lightweight encoder-only weight file next to the full checkpoint (same stem).
    if also_export_encoder:
        enc_path = path.with_name(
            "encoder_best.pt" if path.name.startswith("checkpoint_best") else "encoder_last.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "run_name": cfg.run_name,
                "backbone": cfg.backbone,
                "embed_dim": cfg.embed_dim,
                "encoder": model.encoder.state_dict(),
            },
            enc_path,
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
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"results dir: {out.resolve()}")
    print(f"  checkpoints → checkpoint_last.pt / checkpoint_best.pt")
    print(f"  encoder weights → encoder_last.pt / encoder_best.pt")

    resume_path = Path(cfg.resume) if cfg.resume else None
    ckpt: dict | None = None
    start_epoch = 1
    history: list[dict] = []
    elapsed_before = 0.0
    extend_epochs = False

    if resume_path is not None:
        ckpt = load_training_checkpoint(resume_path, device)
        validate_resume_config(cfg, ckpt["config"])
        start_epoch = int(ckpt["epoch"]) + 1
        history = list(ckpt.get("history", []))
        elapsed_before = float(ckpt.get("elapsed_sec", 0.0))
        saved_epochs = int(ckpt["config"].get("epochs", cfg.epochs))
        extend_epochs = cfg.epochs > saved_epochs
        if start_epoch > cfg.epochs:
            raise ValueError(
                f"Checkpoint epoch {ckpt['epoch']} >= target epochs {cfg.epochs}; nothing to resume"
            )
        if extend_epochs:
            print(
                f"Resuming from {resume_path} at epoch {start_epoch}/{cfg.epochs} "
                f"(extended from {saved_epochs} epochs; fresh cosine LR for remaining steps)"
            )
        else:
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
        f"jepa_infonce_weight={cfg.jepa_infonce_weight} | "
        f"aug_infonce_weight={cfg.aug_infonce_weight} | "
        f"infonce_temperature={cfg.infonce_temperature} | "
        f"rank_weight={cfg.rank_weight} | "
        f"rank_similarity={cfg.rank_similarity} | "
        f"rank_transparency={cfg.rank_transparency} | "
        f"margin={cfg.margin} | "
        f"vicreg_inv_weight={cfg.vicreg_inv_weight} | "
        f"vicreg_var_weight={cfg.vicreg_var_weight} | "
        f"vicreg_cov_weight={cfg.vicreg_cov_weight} | "
        f"sigreg_weight={cfg.sigreg_weight} | "
        f"sigreg_inv_weight={cfg.sigreg_inv_weight} | "
        f"sigreg_num_slices={cfg.sigreg_num_slices} | "
        f"uniformity_weight={cfg.uniformity_weight} | "
        f"uniformity_t={cfg.uniformity_t} | "
        f"proto_weight={cfg.proto_weight} | "
        f"proto_num={cfg.proto_num}"
    )
    if cfg.training_mode == "jepa_gated_centroid_nn":
        print(
            f"centroid_nn mode={cfg.nn_mode} | num_augs={cfg.nn_num_augs} | "
            f"weight={cfg.nn_weight} | intra={cfg.nn_intra_weight} | "
            f"mutual={cfg.nn_mutual} | unif={cfg.uniformity_weight} | "
            f"precompute={cfg.nn_precompute_augs} | jepa=off ema=off"
        )
        if cfg.nn_mode == "pull_push":
            print(f"  pull_push λ_push={cfg.nn_push_weight}")
        elif cfg.nn_mode == "softmax":
            print(f"  softmax T={cfg.nn_temperature}")
        else:
            print(
                f"  gated tau={cfg.nn_tau}→{cfg.nn_tau_end} over {cfg.nn_tau_anneal_epochs} ep | "
                f"k={cfg.nn_k} | soft={cfg.nn_soft_weight} | sim_gap={cfg.nn_sim_gap}"
            )

    if cfg.backbone.lower() in {"vit", "small_vit"}:
        print(
            f"vit_patch_size={cfg.vit_patch_size} | vit_depth={cfg.vit_depth} | "
            f"vit_heads={cfg.vit_heads} | vit_mlp_dim={cfg.vit_mlp_dim} | "
            f"embed_dim={cfg.embed_dim}"
        )

    # For gated centroid-NN: batch_size may be given as total views (e.g. 768 = 128×6).
    unique_batch = cfg.batch_size
    if (
        cfg.training_mode == "jepa_gated_centroid_nn"
        and cfg.nn_num_augs > 1
        and cfg.batch_size % cfg.nn_num_augs == 0
        and cfg.batch_size >= cfg.nn_num_augs * 16
    ):
        unique_batch = cfg.batch_size // cfg.nn_num_augs
        print(
            f"batch_size={cfg.batch_size} interpreted as view-batch → "
            f"{unique_batch} unique images × {cfg.nn_num_augs} augs = {cfg.batch_size} views"
        )

    precomputed_aug_bank = None
    if cfg.training_mode == "jepa_gated_centroid_nn" and cfg.nn_precompute_augs:
        from mask_path_jepa.data import DATASETS, precompute_centroid_aug_bank

        print("Loading or precomputing centroid augs (disk-cached)...")
        precomputed_aug_bank = precompute_centroid_aug_bank(
            cfg.dataset,
            cfg.data_dir,
            num_augs=cfg.nn_num_augs,
            spec=DATASETS[cfg.dataset],
            download=cfg.download,
            train_subset=cfg.train_subset,
            seed=cfg.seed,
            brightness=cfg.aug_brightness,
            contrast=cfg.aug_contrast,
            saturation=cfg.aug_saturation,
            hue=cfg.aug_hue,
            cache_dir=cfg.nn_aug_cache_dir,
            force_recompute=cfg.nn_aug_force_recompute,
        )

    train_loader, test_loader, spec = get_dataloaders(
        cfg.dataset,
        cfg.data_dir,
        unique_batch,
        cfg.num_workers,
        cfg.train_subset,
        download=cfg.download,
        return_index=cfg.teacher_features_path is not None,
        precomputed_aug_bank=precomputed_aug_bank,
    )

    teacher_bank: TeacherFeatureBank | None = None
    if cfg.teacher_features_path:
        teacher_bank = TeacherFeatureBank(cfg.teacher_features_path, device=torch.device("cpu"))
        if teacher_bank.meta.get("dataset") not in (None, cfg.dataset):
            raise ValueError(
                f"Teacher features dataset={teacher_bank.meta.get('dataset')!r} "
                f"!= config dataset={cfg.dataset!r}"
            )
        print(f"teacher bank: {teacher_bank.describe()} | align_weight={cfg.teacher_align_weight} | "
              f"anneal epochs {cfg.teacher_anneal_start_epoch}-{cfg.teacher_anneal_end_epoch}")

    model = JEPA(
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
    use_rank_jepa = (
        cfg.training_mode == "latent_distortion_ranking"
        and (cfg.jepa_infonce_weight > 0 or cfg.align_weight > 0)
    )
    if (
        cfg.anchor_mode == "predictor_corrupt"
        and cfg.training_mode
        not in {"latent_distortion_ranking", "jepa_gated_centroid_nn"}
    ):
        train_params += list(model.predictor.parameters())
    elif use_rank_jepa:
        train_params += list(model.predictor.parameters())
    if model.prototype_bank is not None:
        train_params += list(model.prototype_bank.parameters())
    optimizer = optim.AdamW(
        train_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    remaining_epochs = cfg.epochs - (start_epoch - 1) if ckpt is not None else cfg.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_epochs if extend_epochs else cfg.epochs,
    )
    criterion = JEPALoss(
        margin=cfg.margin,
        triplet_weight=cfg.triplet_weight,
        vicreg_inv_weight=cfg.vicreg_inv_weight,
        vicreg_var_weight=cfg.vicreg_var_weight,
        vicreg_cov_weight=cfg.vicreg_cov_weight,
        sigreg_weight=cfg.sigreg_weight,
        sigreg_inv_weight=cfg.sigreg_inv_weight,
        sigreg_num_slices=cfg.sigreg_num_slices,
        uniformity_weight=(
            0.0
            if cfg.training_mode == "jepa_gated_centroid_nn"
            else cfg.uniformity_weight
        ),
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
        jepa_infonce_augment=(cfg.training_mode == "latent_infonce_jepa_augment"),
        jepa_infonce_vicreg=(
            cfg.training_mode in {"latent_infonce_jepa_vicreg", "latent_infonce_jepa_mse_var_cov"}
        ),
        jepa_infonce_sigreg=(cfg.training_mode == "latent_infonce_jepa_sigreg"),
        distortion_ranking=(cfg.training_mode == "latent_distortion_ranking"),
        rank_weight=cfg.rank_weight,
        aug_align_weight=cfg.aug_align_weight,
        align_weight=cfg.align_weight,
        jepa_infonce_weight=cfg.jepa_infonce_weight,
        aug_infonce_weight=cfg.aug_infonce_weight,
        infonce_temperature=cfg.infonce_temperature,
    )

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if not extend_epochs:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr

    best_knn, best_epoch = best_knn_from_history(history)

    t0 = time.time()
    use_vicreg = (
        cfg.vicreg_inv_weight > 0
        or cfg.vicreg_var_weight > 0
        or cfg.vicreg_cov_weight > 0
    )
    use_sigreg = cfg.sigreg_weight > 0
    use_uniformity = cfg.uniformity_weight > 0 and cfg.training_mode != "jepa_gated_centroid_nn"
    use_gated_centroid_nn = cfg.training_mode == "jepa_gated_centroid_nn"
    use_proto = cfg.proto_weight > 0
    use_triplet_uniformity = cfg.training_mode == "latent_triplet_uniformity"
    use_jepa_augment = cfg.training_mode == "latent_jepa_augment_uniformity"
    use_infonce_jepa_augment = cfg.training_mode == "latent_infonce_jepa_augment"
    use_infonce_jepa_vicreg = cfg.training_mode in {
        "latent_infonce_jepa_vicreg",
        "latent_infonce_jepa_mse_var_cov",
    }
    use_infonce_jepa_sigreg = cfg.training_mode == "latent_infonce_jepa_sigreg"
    use_distortion_ranking = cfg.training_mode == "latent_distortion_ranking"
    steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, cfg.epochs + 1):
        teacher_w = teacher_weight_at_epoch(
            epoch,
            cfg.teacher_align_weight,
            cfg.teacher_anneal_start_epoch,
            cfg.teacher_anneal_end_epoch,
        )
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
            teacher_bank,
            teacher_w,
        )
        scheduler.step()
        row: dict = {"epoch": epoch, **train_stats, "lr": scheduler.get_last_lr()[0], "teacher_w": teacher_w}
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
        if use_gated_centroid_nn:
            if "nn" in row:
                parts.append(f"nn {row['nn']:.4f}")
                if "nn_weighted" in row:
                    parts.append(f"nn×w {row['nn_weighted']:.4f}")
            if "nn_pull" in row:
                parts.append(f"pull {row['nn_pull']:.4f}")
            if "nn_push" in row:
                parts.append(f"push {row['nn_push']:.4f}")
            if "nn_near_sim" in row:
                parts.append(f"near {row['nn_near_sim']:.4f}")
            if "nn_far_sim" in row:
                parts.append(f"far {row['nn_far_sim']:.4f}")
            if "intra" in row:
                parts.append(f"intra {row['intra']:.4f}")
            if "intra_mean_sim" in row:
                parts.append(f"intra_sim {row['intra_mean_sim']:.4f}")
            if "uniformity" in row:
                parts.append(f"unif {row['uniformity']:.4f}")
            if row.get("nn_temp") is not None and row.get("nn_temp", 0) > 0:
                parts.append(f"T {row['nn_temp']:.3f}")
                if "nn_max_p" in row:
                    parts.append(f"max_p {row['nn_max_p']:.3f}")
                if "nn_entropy" in row:
                    parts.append(f"H {row['nn_entropy']:.3f}")
                if "nn_mean_sim" in row:
                    parts.append(f"E[sim] {row['nn_mean_sim']:.4f}")
            elif "nn_pull" not in row:
                parts.append(
                    f"nn_on {int(row.get('nn_active_steps', 0))}/"
                    f"{int(row.get('nn_steps', 0))} "
                    f"({100.0 * row.get('nn_active_frac', 0.0):.1f}%)"
                )
                parts.append(f"nn_pairs_sum {int(row.get('nn_total_pairs', 0))}")
                parts.append(f"tau {row.get('nn_tau', 0):.4f}")
        elif cfg.triplet_weight > 0 and not use_triplet_uniformity and not use_jepa_augment:
            parts.append(f"jepa {row['jepa']:.4f}")
            parts.append(f"triplet {row['triplet']:.4f}")
        elif use_infonce_jepa_sigreg:
            if "jepa_infonce" in row:
                parts.append(f"jepa_nce {row['jepa_infonce']:.4f}")
            if "sigreg_inv" in row:
                parts.append(f"inv {row['sigreg_inv']:.4f}")
            if "sigreg" in row:
                parts.append(f"sigreg {row['sigreg']:.4f}")
            if "teacher_align" in row:
                parts.append(f"teacher {row['teacher_align']:.4f}")
            if "teacher_w" in row and row["teacher_w"] > 0:
                parts.append(f"teacher_w {row['teacher_w']:.3f}")
            if "align" in row:
                parts.append(f"align(log) {row['align']:.4f}")
            if "aug_align" in row:
                parts.append(f"aug_align(log) {row['aug_align']:.4f}")
            if "uniformity" in row:
                parts.append(f"unif {row['uniformity']:.4f}")
            parts.append(f"jepa(log) {row['jepa']:.4f}")
        elif use_infonce_jepa_vicreg:
            if "jepa_infonce" in row:
                parts.append(f"jepa_nce {row['jepa_infonce']:.4f}")
            if "vicreg_inv" in row:
                parts.append(f"inv {row['vicreg_inv']:.4f}")
            if "vicreg_var" in row:
                parts.append(f"var {row['vicreg_var']:.4f}")
            if "vicreg_cov" in row:
                parts.append(f"cov {row['vicreg_cov']:.4f}")
            if "teacher_align" in row:
                parts.append(f"teacher {row['teacher_align']:.4f}")
            if "teacher_w" in row and row["teacher_w"] > 0:
                parts.append(f"teacher_w {row['teacher_w']:.3f}")
            if "align" in row:
                parts.append(f"align(log) {row['align']:.4f}")
            if "aug_align" in row:
                parts.append(f"aug_align(log) {row['aug_align']:.4f}")
            if "uniformity" in row:
                parts.append(f"unif {row['uniformity']:.4f}")
            parts.append(f"jepa(log) {row['jepa']:.4f}")
        elif use_infonce_jepa_augment:
            if "jepa_infonce" in row:
                parts.append(f"jepa_nce {row['jepa_infonce']:.4f}")
            if "aug_infonce" in row:
                parts.append(f"aug_nce {row['aug_infonce']:.4f}")
            if "teacher_align" in row:
                parts.append(f"teacher {row['teacher_align']:.4f}")
            if "teacher_w" in row and row["teacher_w"] > 0:
                parts.append(f"teacher_w {row['teacher_w']:.3f}")
            if "align" in row:
                parts.append(f"align(log) {row['align']:.4f}")
            if "aug_align" in row:
                parts.append(f"aug_align(log) {row['aug_align']:.4f}")
            if "uniformity" in row:
                parts.append(f"unif {row['uniformity']:.4f}")
            parts.append(f"jepa(log) {row['jepa']:.4f}")
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
        elif use_distortion_ranking:
            if "rank" in row:
                parts.append(f"rank {row['rank']:.4f}")
            if "jepa_infonce" in row:
                parts.append(f"jepa_nce {row['jepa_infonce']:.4f}")
            if "aug_infonce" in row:
                parts.append(f"aug_nce {row['aug_infonce']:.4f}")
            if "rank_violation" in row:
                parts.append(f"viol {row['rank_violation']:.4f}")
            for i in range(1, 4):
                key = f"rank_d{i}"
                if key in row:
                    parts.append(f"d{i} {row[key]:.4f}")
        else:
            parts.append(f"jepa {row['jepa']:.4f}")
            if "nn" in row:
                parts.append(f"nn {row['nn']:.4f}")
                if "nn_weighted" in row:
                    parts.append(f"nn×w {row['nn_weighted']:.4f}")
                if "uniformity" in row:
                    parts.append(f"unif {row['uniformity']:.4f}")
                if row.get("nn_temp") is not None and row.get("nn_temp", 0) > 0:
                    parts.append(f"T {row['nn_temp']:.3f}")
                    if "nn_max_p" in row:
                        parts.append(f"max_p {row['nn_max_p']:.3f}")
                    if "nn_entropy" in row:
                        parts.append(f"H {row['nn_entropy']:.3f}")
                    if "nn_mean_sim" in row:
                        parts.append(f"E[sim] {row['nn_mean_sim']:.4f}")
                else:
                    parts.append(
                        f"nn_on {int(row.get('nn_active_steps', 0))}/"
                        f"{int(row.get('nn_steps', 0))} "
                        f"({100.0 * row.get('nn_active_frac', 0.0):.1f}%)"
                    )
                    parts.append(f"nn_pairs_sum {int(row.get('nn_total_pairs', 0))}")
                    parts.append(f"tau {row.get('nn_tau', 0):.4f}")
            else:
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
            also_export_encoder=True,
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
                also_export_encoder=True,
            )
            print(
                f"  -> new best k-NN {best_knn:.3f} @ epoch {best_epoch} "
                f"(saved checkpoint_best.pt + encoder_best.pt)"
            )

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
        "sigreg_inv_weight": cfg.sigreg_inv_weight,
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
        "checkpoint_last": str(checkpoint_last_path(out)),
        "checkpoint_best": str(checkpoint_best_path(out)),
        "encoder_last": str(out / "encoder_last.pt"),
        "encoder_best": str(out / "encoder_best.pt"),
        "final": history[-1],
        "history": history,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'results.json'}")
    print(f"models saved under {out.resolve()}")
    return summary
