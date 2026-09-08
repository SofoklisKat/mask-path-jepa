#!/usr/bin/env python3
"""Supervised CIFAR baseline (cross-entropy) for paper comparison."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mask_path_jepa.data import DATASETS
from mask_path_jepa.eval import evaluate_encoder
from mask_path_jepa.models import build_encoder
from mask_path_jepa.train import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised CIFAR baseline")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--run-name", type=str, default="supervised_cifar10_r18")
    p.add_argument("--dataset", type=str, default="cifar10", choices=sorted(DATASETS.keys()))
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--probe-epochs", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--log-every", type=int, default=50)
    # CLI overrides config (so --device cuda:1 is not clobbered by config "device").
    cli = p.parse_args()
    args = argparse.Namespace(**vars(cli))
    if cli.config:
        with open(cli.config) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key):
                setattr(args, key, v)
            # accept legacy "encoder" key as backbone
            if k == "encoder" and not cfg.get("backbone"):
                args.backbone = v
        # re-apply explicitly provided CLI flags
        for action in p._actions:
            dest = action.dest
            if dest in ("help",) or not hasattr(cli, dest):
                continue
            default = action.default
            if getattr(cli, dest) != default:
                setattr(args, dest, getattr(cli, dest))
    return args


def cosine_lr(optimizer, epoch: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if epoch < warmup:
        lr = base_lr * float(epoch + 1) / float(max(1, warmup))
    else:
        t = (epoch - warmup) / float(max(1, total - warmup))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


class SupervisedNet(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.head(z), z


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.dataset not in {"cifar10", "cifar100"}:
        raise ValueError(f"Supervised baseline supports cifar10/cifar100, got {args.dataset}")
    spec = DATASETS[args.dataset]

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    mean, std = spec.mean, spec.std
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(spec.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    ds_cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    train_set = ds_cls(args.data_dir, train=True, transform=train_tf, download=args.download)
    probe_train = ds_cls(args.data_dir, train=True, transform=eval_tf, download=False)
    test_set = ds_cls(args.data_dir, train=False, transform=eval_tf, download=args.download)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    probe_train_loader = DataLoader(
        probe_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    encoder = build_encoder(
        args.backbone,
        in_channels=spec.in_channels,
        embed_dim=args.embed_dim,
        image_size=spec.image_size,
    )
    model = SupervisedNet(encoder, args.embed_dim, spec.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    history: list[dict] = []
    best_acc = -1.0
    best_knn = -1.0
    best_knn_epoch = -1
    best_knn_test_acc = -1.0
    best_knn_linear = -1.0
    t0 = time.time()
    print(
        f"Supervised baseline | {args.dataset} | {args.backbone} | "
        f"{args.epochs} epochs | {device}"
    )

    for epoch in range(args.epochs):
        model.train()
        lr = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr, args.min_lr)
        loss_sum = 0.0
        acc_sum = 0.0
        n_seen = 0
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            with torch.no_grad():
                batch_acc = (logits.argmax(1) == labels).float().mean().item()
            loss_sum += loss.item() * bs
            acc_sum += batch_acc * bs
            n_seen += bs
            if step % args.log_every == 0:
                print(
                    f"ep {epoch:03d} step {step:04d} | "
                    f"loss {loss_sum / n_seen:.4f} train_acc {acc_sum / n_seen:.4f} lr {lr:.5f}"
                )

        row: dict = {
            "epoch": epoch + 1,  # 1-indexed to match SSL history
            "loss": loss_sum / max(1, n_seen),
            "train_acc": acc_sum / max(1, n_seen),
            "lr": lr,
        }

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            test_acc = evaluate_accuracy(model, test_loader, device)
            metrics = evaluate_encoder(
                model.encoder,
                probe_train_loader,
                test_loader,
                spec.num_classes,
                device,
                knn_k=args.knn_k,
                probe_epochs=args.probe_epochs,
            )
            knn = float(metrics["knn"])
            lin = float(metrics["linear_probe"])
            row.update({"test_acc": test_acc, "knn": knn, "linear_probe": lin})
            print(
                f"==> ep {epoch + 1:03d} | test_acc {test_acc:.4f} | "
                f"knn {knn:.4f} | frozen_linear {lin:.4f}"
            )

            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
                "history": history + [row],
            }
            torch.save(ckpt, out_dir / "checkpoint_last.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "run_name": args.run_name,
                    "backbone": args.backbone,
                    "embed_dim": args.embed_dim,
                    "encoder": model.encoder.state_dict(),
                },
                out_dir / "encoder_last.pt",
            )

            if test_acc > best_acc:
                best_acc = test_acc
                ckpt["best_acc"] = best_acc
                torch.save(ckpt, out_dir / "checkpoint_best.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "run_name": args.run_name,
                        "backbone": args.backbone,
                        "embed_dim": args.embed_dim,
                        "encoder": model.encoder.state_dict(),
                    },
                    out_dir / "encoder_best.pt",
                )

            # Paper rule: pick best k-NN checkpoint; report Lin/test_acc at same epoch.
            if knn > best_knn:
                best_knn = knn
                best_knn_epoch = epoch + 1
                best_knn_test_acc = test_acc
                best_knn_linear = lin

        history.append(row)
        with open(out_dir / "results.json", "w") as f:
            json.dump(
                {
                    "history": history,
                    "best_test_acc": best_acc if best_acc >= 0 else None,
                    "best_knn": best_knn if best_knn >= 0 else None,
                    "best_knn_epoch": best_knn_epoch if best_knn_epoch >= 0 else None,
                    "best_knn_test_acc": best_knn_test_acc if best_knn_test_acc >= 0 else None,
                    "best_knn_linear_probe": best_knn_linear if best_knn_linear >= 0 else None,
                    "elapsed_sec": time.time() - t0,
                },
                f,
                indent=2,
            )

    print(
        f"Done. best_test_acc={best_acc:.4f} best_knn={best_knn:.4f} "
        f"@ep {best_knn_epoch} (test_acc={best_knn_test_acc:.4f}, "
        f"frozen_lin={best_knn_linear:.4f}) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
