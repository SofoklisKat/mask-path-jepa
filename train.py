#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tripletjepa.data import DATASETS
from tripletjepa.train import TrainConfig, run_training


def dataset_choices() -> list[str]:
    return sorted(DATASETS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Triplet-JEPA")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--dataset", type=str, choices=dataset_choices())
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--triplet-weight", type=float)
    parser.add_argument("--align-weight", type=float, help="JEPA align weight (0 = disable)")
    parser.add_argument("--aug-align-weight", type=float, help="Encoder aug vs clean cosine weight")
    parser.add_argument("--jepa-infonce-weight", type=float, help="InfoNCE weight on predictor(mask) vs clean")
    parser.add_argument("--aug-infonce-weight", type=float, help="InfoNCE weight on encoder(aug) vs clean")
    parser.add_argument("--infonce-temperature", type=float, help="Softmax temperature for InfoNCE")
    parser.add_argument("--teacher-features-path", type=str, help="Precomputed .pt from extract_teacher_features.py")
    parser.add_argument("--teacher-align-weight", type=float, help="Weight for teacher-guided batch alignment")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet", "resnet18", "resnet50", "vit", "vit_tiny"],
        help="Encoder backbone: resnet, resnet18, resnet50, vit (custom), or vit_tiny (DeiT-Ti)",
    )
    parser.add_argument("--vit-depth", type=int)
    parser.add_argument("--vit-heads", type=int)
    parser.add_argument("--vit-patch-size", type=int)
    parser.add_argument("--vit-mlp-dim", type=int)
    parser.add_argument(
        "--negative-mode",
        type=str,
        choices=["instance", "class", "scramble", "scramble_class"],
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["jepa_ema", "jepa_gated_centroid_nn", "latent_triplet", "latent_vicreg", "latent_sigreg", "latent_uniformity", "latent_triplet_uniformity", "latent_jepa_augment_uniformity", "latent_infonce_jepa_augment", "latent_infonce_jepa_vicreg", "latent_infonce_jepa_mse_var_cov", "latent_infonce_jepa_sigreg", "latent_distortion_ranking"],
    )
    parser.add_argument("--use-ema-target", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--anchor-mode",
        type=str,
        choices=["predictor_corrupt", "encoder_corrupt", "encoder_clean"],
    )
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument(
        "--device",
        type=str,
        help="Training device: auto, cpu, cuda, cuda:0, cuda:1, ...",
    )
    parser.add_argument("--train-subset", type=int, help="Use first N train samples (debug)")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets if missing (use on server; off by default)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint_last.pt (or other .pt) to resume training",
    )
    args = parser.parse_args()

    if args.config:
        cfg = TrainConfig.from_json(args.config)
    else:
        cfg = TrainConfig()

    if args.run_name:
        cfg.run_name = args.run_name
    if args.dataset:
        cfg.dataset = args.dataset
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.triplet_weight is not None:
        cfg.triplet_weight = args.triplet_weight
    if args.align_weight is not None:
        cfg.align_weight = args.align_weight
    if args.aug_align_weight is not None:
        cfg.aug_align_weight = args.aug_align_weight
    if args.jepa_infonce_weight is not None:
        cfg.jepa_infonce_weight = args.jepa_infonce_weight
    if args.aug_infonce_weight is not None:
        cfg.aug_infonce_weight = args.aug_infonce_weight
    if args.infonce_temperature is not None:
        cfg.infonce_temperature = args.infonce_temperature
    if args.teacher_features_path:
        cfg.teacher_features_path = args.teacher_features_path
    if args.teacher_align_weight is not None:
        cfg.teacher_align_weight = args.teacher_align_weight
    if args.backbone:
        cfg.backbone = args.backbone
    if args.vit_depth is not None:
        cfg.vit_depth = args.vit_depth
    if args.vit_heads is not None:
        cfg.vit_heads = args.vit_heads
    if args.vit_patch_size is not None:
        cfg.vit_patch_size = args.vit_patch_size
    if args.vit_mlp_dim is not None:
        cfg.vit_mlp_dim = args.vit_mlp_dim
    if args.negative_mode:
        cfg.negative_mode = args.negative_mode
    if args.training_mode:
        cfg.training_mode = args.training_mode
    if args.use_ema_target is not None:
        cfg.use_ema_target = args.use_ema_target
    if args.anchor_mode:
        cfg.anchor_mode = args.anchor_mode
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device
    if args.train_subset is not None:
        cfg.train_subset = args.train_subset
    if args.download:
        cfg.download = True
    if args.resume:
        cfg.resume = args.resume

    summary = run_training(cfg)
    print(json.dumps(summary["final"], indent=2))


if __name__ == "__main__":
    main()
