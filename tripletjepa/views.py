from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
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


def gaussian_blur(
    x: torch.Tensor,
    sigma: float,
    *,
    max_kernel: int | None = None,
) -> torch.Tensor:
    """Isotropic Gaussian blur; no-op when sigma is negligible."""
    if sigma <= 1e-3:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel = 2 * radius + 1
    if max_kernel is not None:
        k = max_kernel if max_kernel % 2 == 1 else max_kernel - 1
        kernel = min(kernel, max(3, k))
    return TF.gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def blur_patch_tile(patch: torch.Tensor, sigma: float, patch_size: int) -> torch.Tensor:
    """Blur one grid tile; uses local box blur (safe for 4×4 at any curriculum σ).

    ``torchvision.gaussian_blur`` needs padding < tile size, so large σ on 4×4
    patches crashes. Box-kernel size tracks σ instead: σ≈0.5 → 3×3, σ≥1.5 → full tile.
    """
    if sigma <= 1e-3:
        return patch
    max_k = patch_size if patch_size % 2 == 1 else patch_size - 1
    k = min(max_k, 2 * int(math.ceil(sigma)) + 1)
    if k % 2 == 0:
        k -= 1
    k = max(1, k)
    if k <= 1:
        return patch
    pad = k // 2
    padded = F.pad(patch, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=k, stride=1)


def corrupt_progress(epoch: int, epochs: int) -> float:
    """Training progress in [0, 1] for corruption curriculum (epoch 1 → 0, last → 1)."""
    if epochs <= 1:
        return 1.0
    return (epoch - 1) / (epochs - 1)


def blur_patches(
    x: torch.Tensor,
    mask_ratio: float,
    sigma: float,
    *,
    patch_size: int = 4,
) -> torch.Tensor:
    """Blur a random subset of non-overlapping image patches; others stay sharp.

    Corruption is **patch-local**: each selected 4×4 tile is blurred on its own.
    The rest of the image is left unchanged (no whole-image blur).
    """
    if mask_ratio <= 0 or sigma <= 1e-3:
        return x
    b, c, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image size ({h}, {w}) must be divisible by patch_size={patch_size}")
    out = x.clone()
    grid_h, grid_w = h // patch_size, w // patch_size
    n_patches = max(1, int(mask_ratio * grid_h * grid_w))

    for i in range(b):
        positions = torch.randperm(grid_h * grid_w, device=x.device)[:n_patches]
        for pos in positions:
            gh = int(pos // grid_w)
            gw = int(pos % grid_w)
            top, left = gh * patch_size, gw * patch_size
            patch = x[i : i + 1, :, top : top + patch_size, left : left + patch_size]
            out[i : i + 1, :, top : top + patch_size, left : left + patch_size] = blur_patch_tile(
                patch, sigma, patch_size
            )
    return out


def active_patch_blur_ratio(
    progress: float,
    ratio_min: float,
    ratio_max: float,
) -> float:
    """Interpolate patch-blur fraction between ``ratio_min`` and ``ratio_max``."""
    progress = float(max(0.0, min(1.0, progress)))
    return ratio_min + progress * (ratio_max - ratio_min)


def progressive_patch_blur(
    x: torch.Tensor,
    progress: float,
    *,
    mask_ratio_end: float = 0.6,
    mask_ratio_start: float = 0.1,
    sigma_min: float = 0.5,
    sigma_max: float = 3.0,
    patch_size: int = 4,
) -> torch.Tensor:
    """Ramp patch-local blur: more tiles corrupted + stronger σ per tile (not whole-image)."""
    progress = float(max(0.0, min(1.0, progress)))
    patch_ratio = active_patch_blur_ratio(progress, mask_ratio_start, mask_ratio_end)
    sigma = sigma_min + progress * (sigma_max - sigma_min)
    return blur_patches(x, mask_ratio=patch_ratio, sigma=sigma, patch_size=patch_size)


def progressive_block_mask(
    x: torch.Tensor,
    progress: float,
    *,
    mask_ratio_end: float = 0.6,
    mask_ratio_start: float = 0.1,
    min_block: int = 4,
) -> torch.Tensor:
    """Ramp multi-block zero masking: fewer masked blocks early, more later."""
    progress = float(max(0.0, min(1.0, progress)))
    ratio = active_patch_blur_ratio(progress, mask_ratio_start, mask_ratio_end)
    return block_mask(x, mask_ratio=ratio, min_block=min_block)


def progressive_blur_to_mask(
    x: torch.Tensor,
    progress: float,
    *,
    sigma_min: float = 0.5,
    sigma_max: float = 3.0,
    mask_ratio_end: float = 0.6,
    mask_ratio_start: float = 0.1,
    min_block: int = 4,
) -> torch.Tensor:
    """Alias: patch-blur curriculum (kept for config compatibility)."""
    return progressive_patch_blur(
        x,
        progress,
        mask_ratio_end=mask_ratio_end,
        mask_ratio_start=mask_ratio_start,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        patch_size=min_block,
    )


def make_corrupt_view(
    x: torch.Tensor,
    *,
    schedule: str,
    progress: float = 1.0,
    mask_ratio: float = 0.6,
    patch_blur_ratio_min: float = 0.1,
    blur_sigma_min: float = 0.5,
    blur_sigma_max: float = 3.0,
    min_block: int = 4,
) -> torch.Tensor:
    """Build the context (corrupt) view for JEPA training."""
    if schedule == "block":
        return block_mask(x, mask_ratio=mask_ratio, min_block=min_block)
    if schedule == "block_curriculum":
        return progressive_block_mask(
            x,
            progress,
            mask_ratio_end=mask_ratio,
            mask_ratio_start=patch_blur_ratio_min,
            min_block=min_block,
        )
    if schedule == "mask_curriculum":
        return progressive_patch_blur(
            x,
            progress,
            mask_ratio_end=mask_ratio,
            mask_ratio_start=patch_blur_ratio_min,
            sigma_min=blur_sigma_min,
            sigma_max=blur_sigma_max,
            patch_size=min_block,
        )
    if schedule == "blur_to_mask":
        return progressive_blur_to_mask(
            x,
            progress,
            sigma_min=blur_sigma_min,
            sigma_max=blur_sigma_max,
            mask_ratio_end=mask_ratio,
            mask_ratio_start=patch_blur_ratio_min,
            min_block=min_block,
        )
    raise ValueError(
        f"Unknown corrupt_schedule={schedule!r}; "
        "expected 'block', 'block_curriculum', 'mask_curriculum', or 'blur_to_mask'"
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


def make_augmented_view(
    x: torch.Tensor,
    *,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue: float = 0.1,
    p_flip: float = 0.5,
) -> torch.Tensor:
    """Strong appearance aug on tensor views: flip + color jitter (geometry-preserving).

    Applied on top of the dataloader crop/flip so the encoder must use shape/texture,
    not absolute color or low-level edge cues alone.
    """
    out = x.clone()
    for i in range(out.size(0)):
        img = out[i]
        if torch.rand(1, device=x.device).item() < p_flip:
            img = TF.hflip(img)
        b_factor = 1.0 + random.uniform(-brightness, brightness)
        img = TF.adjust_brightness(img, b_factor)
        c_factor = 1.0 + random.uniform(-contrast, contrast)
        img = TF.adjust_contrast(img, c_factor)
        s_factor = 1.0 + random.uniform(-saturation, saturation)
        img = TF.adjust_saturation(img, s_factor)
        h_shift = random.uniform(-hue, hue)
        img = TF.adjust_hue(img, h_shift)
        out[i] = img
    return out


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
