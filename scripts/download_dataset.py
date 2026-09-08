#!/usr/bin/env python3
"""Download datasets for Mask-Path JEPA (run on server or when you want local copies)."""
from __future__ import annotations

import argparse
from pathlib import Path

from mask_path_jepa.data import DATASETS, get_dataloaders


def download_dataset(name: str, data_dir: str) -> None:
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset {name!r}. Choose from {sorted(DATASETS)}")

    print(f"Downloading {name} into {data_dir} ...")
    train_loader, test_loader, spec = get_dataloaders(
        name,
        data_dir,
        batch_size=64,
        num_workers=0,
        download=True,
    )
    print(
        f"Done: {name} | classes={spec.num_classes} | "
        f"train={len(train_loader.dataset)} | test={len(test_loader.dataset)}"
    )
    print(f"Location: {Path(data_dir).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="Dataset to download (e.g. cifar100)",
    )
    parser.add_argument("--data-dir", default="./data", help="Where to store data")
    args = parser.parse_args()
    download_dataset(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
