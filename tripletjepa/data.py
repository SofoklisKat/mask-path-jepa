from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

# Expected local layout under data_dir (no download unless download=True).
DATA_PATH_HINTS: dict[str, str] = {
    "cifar10": "data/cifar-10-batches-py/",
    "cifar100": "data/cifar-100-python/",
    "mnist": "data/MNIST/raw/",
    "tiny_imagenet": "data/tiny-imagenet-200/train/ and data/tiny-imagenet-200/val/",
    "imagenet": "data/imagenet/train/<class>/ and data/imagenet/val/<class>/",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    in_channels: int
    num_classes: int
    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASETS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        "cifar10", 3, 10, 32, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    ),
    "cifar100": DatasetSpec(
        "cifar100", 3, 100, 32, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    ),
    "tiny_imagenet": DatasetSpec(
        "tiny_imagenet",
        3,
        200,
        64,
        (0.4802, 0.4481, 0.3975),
        (0.2302, 0.2265, 0.2262),
    ),
    "imagenet": DatasetSpec(
        "imagenet",
        3,
        1000,
        64,
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    "mnist": DatasetSpec("mnist", 1, 10, 28, (0.1307,), (0.3081,)),
}

CIFAR_LIKE = frozenset({"cifar10", "cifar100"})


def resolve_global_index(base: Dataset, i: int) -> int:
    if isinstance(base, Subset):
        return resolve_global_index(base.dataset, base.indices[i])
    return i


class WithIndex(Dataset):
    """Append global dataset index to each sample."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        image, label = self.base[i]
        return image, label, resolve_global_index(self.base, i)


def build_transforms(spec: DatasetSpec, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(spec.mean, spec.std)
    if spec.name in CIFAR_LIKE:
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(spec.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return transforms.Compose([transforms.ToTensor(), normalize])

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(spec.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(spec.image_size),
            transforms.CenterCrop(spec.image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class TinyImageNetValDataset(Dataset):
    """Validation split (flat images + val_annotations.txt)."""

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        wnids = [line.strip() for line in (root / "wnids.txt").read_text().splitlines()]
        wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        val_dir = root / "val" / "images"
        ann_path = root / "val" / "val_annotations.txt"
        for line in ann_path.read_text().splitlines():
            fname, wnid = line.split("\t")[:2]
            self.samples.append((val_dir / fname, wnid_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))
        return image, label


def _missing_dataset_error(dataset: str, data_dir: Path) -> FileNotFoundError:
    hint = DATA_PATH_HINTS.get(dataset, "see README")
    return FileNotFoundError(
        f"Dataset {dataset!r} not found under {data_dir}. "
        f"Expected: {hint}. "
        "Copy data from your server or run with --download on a machine where downloads are OK."
    )


def _tiny_imagenet_ready(root: Path) -> bool:
    return (root / "train").exists() and (root / "val" / "val_annotations.txt").exists()


def prepare_tiny_imagenet(data_dir: Path, download: bool) -> Path:
    root = data_dir / "tiny-imagenet-200"
    if _tiny_imagenet_ready(root):
        return root
    if not download:
        raise _missing_dataset_error("tiny_imagenet", data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "tiny-imagenet-200.zip"
    if not zip_path.exists():
        print(f"Downloading Tiny ImageNet to {zip_path} ...")
        urlretrieve(TINY_IMAGENET_URL, zip_path)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    if not _tiny_imagenet_ready(root):
        raise RuntimeError(f"Tiny ImageNet extraction failed under {root}")
    return root


def _build_torchvision_dataset(
    dataset: str,
    data_dir: str,
    train: bool,
    transform: transforms.Compose,
    download: bool,
) -> Dataset:
    path = Path(data_dir)
    if dataset == "cifar10":
        return datasets.CIFAR10(path, train=train, download=download, transform=transform)
    if dataset == "cifar100":
        return datasets.CIFAR100(path, train=train, download=download, transform=transform)
    if dataset == "mnist":
        return datasets.MNIST(path, train=train, download=download, transform=transform)
    if dataset == "tiny_imagenet":
        root = prepare_tiny_imagenet(path, download=download)
        if train:
            return datasets.ImageFolder(root / "train", transform=transform)
        return TinyImageNetValDataset(root, transform)
    if dataset == "imagenet":
        split = "train" if train else "val"
        imagenet_root = path / "imagenet" / split
        if not imagenet_root.exists():
            raise _missing_dataset_error("imagenet", path)
        return datasets.ImageFolder(imagenet_root, transform=transform)
    raise ValueError(f"Unsupported dataset {dataset!r}")


def get_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    train_subset: int | None = None,
    download: bool = False,
    return_index: bool = False,
) -> tuple[DataLoader, DataLoader, DatasetSpec]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {sorted(DATASETS)}")
    spec = DATASETS[dataset]

    try:
        train_set = _build_torchvision_dataset(
            dataset, data_dir, train=True, transform=build_transforms(spec, True), download=download
        )
        test_set = _build_torchvision_dataset(
            dataset, data_dir, train=False, transform=build_transforms(spec, False), download=download
        )
    except RuntimeError as exc:
        if not download and "not found" in str(exc).lower():
            raise _missing_dataset_error(dataset, Path(data_dir)) from exc
        raise

    if train_subset is not None:
        train_set = torch.utils.data.Subset(
            train_set, range(min(train_subset, len(train_set)))
        )
    if return_index:
        train_set = WithIndex(train_set)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, spec
