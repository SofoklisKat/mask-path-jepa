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


def vicreg_invariance(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """MSE between two views (VICReg alignment / invariance term)."""
    return F.mse_loss(z_a, z_b)


def vicreg_variance(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Penalize low per-dimension std across the batch (anti-collapse)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def vicreg_covariance(z: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal covariance entries (decorrelate latent dims)."""
    z = z - z.mean(dim=0)
    n = z.size(0)
    denom = max(n - 1, 1)
    cov = (z.T @ z) / denom
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / z.size(1)


@dataclass
class TripletJEPALoss:
    margin: float = 0.2
    triplet_weight: float = 0.5
    vicreg_inv_weight: float = 0.0
    vicreg_var_weight: float = 0.0
    vicreg_cov_weight: float = 0.0

    def __call__(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor | None = None,
        z_negative_extra: torch.Tensor | None = None,
        z_vicreg_a: torch.Tensor | None = None,
        z_vicreg_b: torch.Tensor | None = None,
        z_inv_a: torch.Tensor | None = None,
        z_inv_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_jepa = jepa_cosine_loss(z_anchor, z_positive)
        stats: dict[str, float] = {"jepa": l_jepa.item()}
        use_vicreg = (
            self.vicreg_inv_weight > 0
            or self.vicreg_var_weight > 0
            or self.vicreg_cov_weight > 0
        )

        if use_vicreg:
            if z_vicreg_a is None or z_vicreg_b is None:
                raise ValueError("z_vicreg_a and z_vicreg_b are required for VICReg")
            total = torch.tensor(0.0, device=z_vicreg_a.device)
            if self.vicreg_inv_weight > 0:
                if z_inv_a is None or z_inv_b is None:
                    raise ValueError("z_inv_a and z_inv_b are required for VICReg invariance")
                l_inv = vicreg_invariance(z_inv_a, z_inv_b)
                total = total + self.vicreg_inv_weight * l_inv
                stats["vicreg_inv"] = l_inv.item()
            if self.vicreg_var_weight > 0:
                l_var = vicreg_variance(z_vicreg_a) + vicreg_variance(z_vicreg_b)
                total = total + self.vicreg_var_weight * l_var
                stats["vicreg_var"] = l_var.item()
            if self.vicreg_cov_weight > 0:
                l_cov = vicreg_covariance(z_vicreg_a) + vicreg_covariance(z_vicreg_b)
                total = total + self.vicreg_cov_weight * l_cov
                stats["vicreg_cov"] = l_cov.item()
        else:
            total = l_jepa

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
            total = total + self.triplet_weight * l_triplet
            stats["triplet"] = l_triplet.item()

        stats["loss"] = total.item()
        return total, stats
