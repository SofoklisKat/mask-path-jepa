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
    """Margin ranking: anchor closer to positive than negative by at least margin."""
    d_pos = (anchor - positive).pow(2).sum(dim=-1)
    d_neg = (anchor - negative).pow(2).sum(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


@dataclass
class TripletJEPALoss:
    margin: float = 0.2
    triplet_weight: float = 0.5

    def __call__(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_jepa = jepa_cosine_loss(z_anchor, z_positive)
        l_triplet = triplet_loss(z_anchor, z_positive, z_negative, margin=self.margin)
        total = l_jepa + self.triplet_weight * l_triplet
        return total, {
            "loss": total.item(),
            "jepa": l_jepa.item(),
            "triplet": l_triplet.item(),
        }
