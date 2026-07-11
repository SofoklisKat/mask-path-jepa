from __future__ import annotations

import torch


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
