from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def jepa_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Regression in latent space: pull predictor output toward target embedding."""
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Margin ranking on L2-normalized embeddings (geometry, not vector scale)."""
    anchor_n = F.normalize(anchor, dim=-1)
    positive_n = F.normalize(positive, dim=-1)
    negative_n = F.normalize(negative, dim=-1)
    d_pos = (anchor_n - positive_n).pow(2).sum(dim=-1)
    d_neg = (anchor_n - negative_n).pow(2).sum(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


@dataclass
class TripletJEPALoss:
    margin: float = 0.2
    triplet_weight: float = 0.5

    def __call__(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor | None = None,
        z_negative_extra: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_jepa = jepa_cosine_loss(z_anchor, z_positive)
        stats: dict[str, float] = {
            "loss": 0.0,  # filled below
            "jepa": l_jepa.item(),
        }
        if self.triplet_weight > 0:
            if z_negative is None:
                raise ValueError("z_negative is required when triplet_weight > 0")
            l_triplet = triplet_loss(z_anchor, z_positive, z_negative, margin=self.margin)
            if z_negative_extra is not None:
                l_extra = triplet_loss(
                    z_anchor, z_positive, z_negative_extra, margin=self.margin
                )
                stats["triplet_scramble"] = l_triplet.item()
                stats["triplet_class"] = l_extra.item()
                l_triplet = 0.5 * (l_triplet + l_extra)
            total = l_jepa + self.triplet_weight * l_triplet
            stats["triplet"] = l_triplet.item()
        else:
            total = l_jepa
        stats["loss"] = total.item()
        return total, stats
