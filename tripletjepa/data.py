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


class PrecomputedMultiAugDataset(Dataset):
    """Train set with live JEPA clean view + fixed precomputed NN aug stack.

    Each item returns ``(clean, label, augs)`` where ``augs`` has shape ``(M, C, H, W)``
    (already normalized). Augmentations are built once before training.
    """

    def __init__(
        self,
        clean_base: Dataset,
        aug_bank: torch.Tensor,
    ) -> None:
        if aug_bank.ndim != 5:
            raise ValueError(f"aug_bank must be (N, M, C, H, W), got {tuple(aug_bank.shape)}")
        if len(clean_base) != aug_bank.size(0):
            raise ValueError(
                f"clean_base len {len(clean_base)} != aug_bank N {aug_bank.size(0)}"
            )
        self.clean_base = clean_base
        self.aug_bank = aug_bank  # float32 CPU, normalized

    def __len__(self) -> int:
        return len(self.clean_base)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        image, label = self.clean_base[i]
        return image, int(label), self.aug_bank[i]


def _pil_strong_aug_to_tensor(
    pil: Image.Image,
    *,
    image_size: int,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    crop_scale_min: float,
    max_rotate_deg: float,
    noise_std: float,
    p_mask: float,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    rng: torch.Generator,
) -> torch.Tensor:
    """One strong aug → normalized CHW float tensor (CPU)."""
    import random as _random

    # Use python random seeded from torch generator for variety
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
    _random.seed(seed)
    t = transforms.ToTensor()(pil)  # [0,1]
    c, h, w = t.shape
    assert h == image_size and w == image_size

    scale = _random.uniform(crop_scale_min, 1.0)
    ch = max(1, int(round(h * scale)))
    cw = max(1, int(round(w * scale)))
    top = _random.randint(0, h - ch) if ch < h else 0
    left = _random.randint(0, w - cw) if cw < w else 0
    t = transforms.functional.resized_crop(t, top, left, ch, cw, size=[h, w])
    if max_rotate_deg > 0:
        angle = _random.uniform(-max_rotate_deg, max_rotate_deg)
        t = transforms.functional.rotate(
            t, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0.0
        )
    if _random.random() < 0.5:
        t = transforms.functional.hflip(t)
    t = transforms.functional.adjust_brightness(t, 1.0 + _random.uniform(-brightness, brightness))
    t = transforms.functional.adjust_contrast(t, 1.0 + _random.uniform(-contrast, contrast))
    t = transforms.functional.adjust_saturation(t, 1.0 + _random.uniform(-saturation, saturation))
    t = transforms.functional.adjust_hue(t, _random.uniform(-hue, hue))
    if _random.random() < p_mask:
        mh = _random.randint(4, max(4, h // 2))
        mw = _random.randint(4, max(4, w // 2))
        mt = _random.randint(0, h - mh)
        ml = _random.randint(0, w - mw)
        fill = t.mean(dim=(1, 2), keepdim=True)
        t[:, mt : mt + mh, ml : ml + mw] = fill
    if noise_std > 0:
        t = t + torch.randn(t.shape, generator=rng) * noise_std
    t = t.clamp(0.0, 1.0)
    t = transforms.Normalize(mean, std)(t)
    return t


def centroid_aug_bank_cache_path(
    cache_dir: str | Path,
    *,
    dataset_name: str,
    num_augs: int,
    seed: int,
    train_subset: int | None,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    image_size: int,
) -> Path:
    """Deterministic cache filename for a precomputed aug bank."""
    import hashlib
    import json

    meta = {
        "dataset": dataset_name,
        "num_augs": int(num_augs),
        "seed": int(seed),
        "train_subset": train_subset,
        "brightness": float(brightness),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "hue": float(hue),
        "image_size": int(image_size),
        "crop_scale_min": 0.7,
        "max_rotate_deg": 15.0,
        "noise_std": 0.03,
        "p_mask": 0.5,
        "version": 1,
    }
    digest = hashlib.sha1(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    subset_tag = "full" if train_subset is None else f"n{train_subset}"
    name = f"{dataset_name}_m{num_augs}_{subset_tag}_seed{seed}_{digest}.pt"
    return Path(cache_dir) / name


def precompute_centroid_aug_bank(
    dataset_name: str,
    data_dir: str,
    *,
    num_augs: int,
    spec: DatasetSpec,
    download: bool = False,
    train_subset: int | None = None,
    seed: int = 42,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue: float = 0.1,
    show_progress: bool = True,
    cache_dir: str | Path | None = None,
    force_recompute: bool = False,
) -> torch.Tensor:
    """Precompute (N, M, C, H, W) normalized aug bank (CPU float32).

    If ``cache_dir`` is set, load from disk when a matching cache exists; otherwise
    build and save so later experiments reuse the same augs.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = centroid_aug_bank_cache_path(
            cache_dir,
            dataset_name=dataset_name,
            num_augs=num_augs,
            seed=seed,
            train_subset=train_subset,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            image_size=spec.image_size,
        )
        if cache_path.is_file() and not force_recompute:
            if show_progress:
                print(f"Loading cached aug bank: {cache_path}")
            obj = torch.load(cache_path, map_location="cpu", weights_only=False)
            bank = obj["bank"] if isinstance(obj, dict) and "bank" in obj else obj
            if not torch.is_tensor(bank):
                raise TypeError(f"Invalid aug bank cache at {cache_path}")
            if show_progress:
                print(f"loaded aug bank shape={tuple(bank.shape)} dtype={bank.dtype}")
            return bank

    raw = _build_torchvision_dataset(
        dataset_name, data_dir, train=True, transform=None, download=download
    )
    if train_subset is not None:
        raw = Subset(raw, range(min(train_subset, len(raw))))

    n = len(raw)
    c, h = spec.in_channels, spec.image_size
    bank = torch.empty(n, num_augs, c, h, h, dtype=torch.float32)
    rng = torch.Generator()
    rng.manual_seed(seed)

    for i in range(n):
        item = raw[i]
        pil = item[0] if not isinstance(item[0], torch.Tensor) else transforms.ToPILImage()(item[0])
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(pil) if hasattr(pil, "shape") else pil
        for m in range(num_augs):
            bank[i, m] = _pil_strong_aug_to_tensor(
                pil,
                image_size=spec.image_size,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                crop_scale_min=0.7,
                max_rotate_deg=15.0,
                noise_std=0.03,
                p_mask=0.5,
                mean=spec.mean,
                std=spec.std,
                rng=rng,
            )
        if show_progress and (i + 1) % 5000 == 0:
            print(f"  precompute augs {i + 1}/{n}")
    if show_progress:
        print(f"precomputed aug bank shape={tuple(bank.shape)} dtype={bank.dtype}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bank": bank,
            "meta": {
                "dataset": dataset_name,
                "num_augs": num_augs,
                "seed": seed,
                "train_subset": train_subset,
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "hue": hue,
                "shape": list(bank.shape),
            },
        }
        torch.save(payload, cache_path)
        if show_progress:
            print(f"Saved aug bank cache: {cache_path}")
    return bank


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
    transform: transforms.Compose | None,
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
    precomputed_aug_bank: torch.Tensor | None = None,
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
    if precomputed_aug_bank is not None:
        if return_index:
            raise ValueError("precomputed_aug_bank cannot be combined with return_index")
        train_set = PrecomputedMultiAugDataset(train_set, precomputed_aug_bank)
    elif return_index:
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
