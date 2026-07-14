from __future__ import annotations

import math

import torch
import torchvision.transforms.functional as TF


def block_mask(
    x: torch.Tensor,
    mask_ratio: float = 0.6,
    min_block: int = 4,
) -> torch.Tensor:
    """Multi-block masking inspired by I-JEPA (semantic blocks on 32x32)."""
    b, c, h, w = x.shape
    out = x.clone()
    block = max(h // 4, min_block)
    grid_h, grid_w = h // block, w // block
    n_blocks = max(1, int(mask_ratio * grid_h * grid_w))

    for i in range(b):
        positions = torch.randperm(grid_h * grid_w)[:n_blocks]
        for pos in positions:
            gh = int(pos // grid_w)
            gw = int(pos % grid_w)
            top, left = gh * block, gw * block
            out[i, :, top : top + block, left : left + block] = 0.0
    return out


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Isotropic Gaussian blur; no-op when sigma is negligible."""
    if sigma <= 1e-3:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel = 2 * radius + 1
    return TF.gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def corrupt_progress(epoch: int, epochs: int) -> float:
    """Training progress in [0, 1] for corruption curriculum (epoch 1 → 0, last → 1)."""
    if epochs <= 1:
        return 1.0
    return (epoch - 1) / (epochs - 1)


def progressive_mask(
    x: torch.Tensor,
    progress: float,
    *,
    mask_ratio_end: float = 0.6,
    min_block: int = 4,
) -> torch.Tensor:
    """Ramp block-mask ratio from 0 (clean context) to ``mask_ratio_end`` (CBM-style)."""
    progress = float(max(0.0, min(1.0, progress)))
    mask_ratio = progress * mask_ratio_end
    if mask_ratio <= 0:
        return x
    return block_mask(x, mask_ratio=mask_ratio, min_block=min_block)


def progressive_blur_to_mask(
    x: torch.Tensor,
    progress: float,
    *,
    sigma_min: float = 0.5,
    sigma_max: float = 3.0,
    mask_ratio_end: float = 0.6,
    min_block: int = 4,
) -> torch.Tensor:
    """Ramp corrupt views from light blur to heavy blur + block masking.

    progress=0: slight blur only (no masking).
    progress=1: strong blur with ``mask_ratio_end`` area zeroed (I-JEPA blocks).
    """
    progress = float(max(0.0, min(1.0, progress)))
    sigma = sigma_min + progress * (sigma_max - sigma_min)
    mask_ratio = progress * mask_ratio_end
    out = gaussian_blur(x, sigma)
    if mask_ratio > 0:
        out = block_mask(out, mask_ratio=mask_ratio, min_block=min_block)
    return out


def make_corrupt_view(
    x: torch.Tensor,
    *,
    schedule: str,
    progress: float = 1.0,
    mask_ratio: float = 0.6,
    blur_sigma_min: float = 0.5,
    blur_sigma_max: float = 3.0,
    min_block: int = 4,
) -> torch.Tensor:
    """Build the context (corrupt) view for JEPA training."""
    if schedule == "block":
        return block_mask(x, mask_ratio=mask_ratio, min_block=min_block)
    if schedule == "mask_curriculum":
        return progressive_mask(
            x,
            progress,
            mask_ratio_end=mask_ratio,
            min_block=min_block,
        )
    if schedule == "blur_to_mask":
        return progressive_blur_to_mask(
            x,
            progress,
            sigma_min=blur_sigma_min,
            sigma_max=blur_sigma_max,
            mask_ratio_end=mask_ratio,
            min_block=min_block,
        )
    raise ValueError(
        f"Unknown corrupt_schedule={schedule!r}; "
        "expected 'block', 'mask_curriculum', or 'blur_to_mask'"
    )


def scramble_patches(
    x: torch.Tensor,
    patch_size: int = 4,
) -> torch.Tensor:
    """Shuffle grid patches in-place to destroy global spatial context.

    Same local patch content as the source image, but permuted positions act as a
    hard negative view for triplet margin (complements block_mask corrupt views).
    """
    b, c, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image size ({h}, {w}) must be divisible by patch_size={patch_size}")
    gh, gw = h // patch_size, w // patch_size
    n = gh * gw

    patches = x.view(b, c, gh, patch_size, gw, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, n, patch_size, patch_size)

    perm = torch.argsort(torch.rand(b, n, device=x.device), dim=1)
    idx = perm[:, None, :, None, None].expand(b, c, n, patch_size, patch_size)
    shuffled = patches.gather(2, idx)

    shuffled = shuffled.view(b, c, gh, gw, patch_size, patch_size)
    shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).contiguous()
    return shuffled.view(b, c, h, w)


def instance_negatives(z: torch.Tensor) -> torch.Tensor:
    """Unsupervised negative: shuffle batch so z-[i] is another image's embedding."""
    idx = torch.randperm(z.size(0), device=z.device)
    # Avoid self-matches when batch has duplicate ordering.
    if (idx == torch.arange(z.size(0), device=z.device)).all():
        idx = torch.roll(idx, shifts=1)
    return z[idx]


def class_negatives(
    z: torch.Tensor,
    labels: torch.Tensor,
    max_tries: int = 8,
) -> torch.Tensor:
    """Sample negatives from a different class when labels are available."""
    b = z.size(0)
    neg_idx = torch.arange(b, device=z.device)
    for i in range(b):
        for _ in range(max_tries):
            j = torch.randint(0, b, (1,), device=z.device).item()
            if labels[j] != labels[i]:
                neg_idx[i] = j
                break
        else:
            # Fallback if batch is class-homogeneous (rare with large batches).
            neg_idx[i] = (i + 1) % b
    return z[neg_idx]
