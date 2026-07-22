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


def triplet_cosine_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Margin triplet on unit sphere using JEPA-style cosine distance (1 - cos)."""
    anchor_n = F.normalize(anchor, dim=-1)
    positive_n = F.normalize(positive, dim=-1)
    negative_n = F.normalize(negative, dim=-1)
    d_pos = 1.0 - (anchor_n * positive_n).sum(dim=-1)
    d_neg = 1.0 - (anchor_n * negative_n).sum(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


def cosine_distance_from_anchor(anchor: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
    """Cosine distance 1 - cos between anchor (B, D) and views (B, L, D)."""
    anchor_n = F.normalize(anchor, dim=-1)
    views_n = F.normalize(views, dim=-1)
    return 1.0 - (views_n * anchor_n.unsqueeze(1)).sum(dim=-1)


def ranking_cosine_loss(
    anchor: torch.Tensor,
    levels: torch.Tensor,
    margin: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize violations of d(anchor, level_i) < d(anchor, level_{i+1}).

    ``levels`` are increasingly severe corrupt views only (not clean).
    """
    dist = cosine_distance_from_anchor(anchor, levels)
    loss = torch.tensor(0.0, device=anchor.device)
    n_pairs = dist.size(1) - 1
    for i in range(n_pairs):
        loss = loss + F.relu(dist[:, i] - dist[:, i + 1] + margin).mean()
    # Mild corruption should sit at least ``margin`` away from clean (dist=0).
    loss = loss + F.relu(margin - dist[:, 0]).mean()

    stats: dict[str, float] = {"rank": loss.item()}
    with torch.no_grad():
        mean_dist = dist.mean(dim=0)
        for i, d in enumerate(mean_dist.tolist()):
            stats[f"rank_d{i + 1}"] = d
        if n_pairs > 0:
            violations = (dist[:, :-1] - dist[:, 1:] + margin).clamp(min=0.0)
            stats["rank_violation"] = violations.mean().item()
    return loss, stats


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


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Hypersphere uniformity (Wang & Isola): spread unit-normalized embeddings."""
    z = F.normalize(z, dim=-1)
    sq_dist = torch.cdist(z, z, p=2).pow(2)
    mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    return torch.log(torch.exp(-t * sq_dist[mask]).mean() + 1e-8)


def infonce_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SimCLR-style InfoNCE on the unit sphere.

    query[i] should match key[i]; all other keys in the batch are negatives.
    Pass key stop-grad (encoder(clean).detach()) for stable JEPA-style training.
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = query @ key.T / temperature
    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def sinkhorn_knopp(
    scores: torch.Tensor,
    n_iters: int = 3,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Soft equipartition assignment (SwAV). scores: (B, K) -> Q: (B, K)."""
    q = torch.exp(scores / epsilon).t()  # (K, B)
    b = q.size(1)
    k = q.size(0)
    q /= torch.sum(q) + 1e-12
    for _ in range(n_iters):
        q /= torch.sum(q, dim=1, keepdim=True) + 1e-12
        q /= k
        q /= torch.sum(q, dim=0, keepdim=True) + 1e-12
        q /= b
    q *= b  # columns sum to 1
    return q.t()


def swav_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.1,
    sinkhorn_iters: int = 3,
    sinkhorn_eps: float = 0.05,
) -> torch.Tensor:
    """SwAV swapped prediction between two unit-sphere views and prototypes (K, D)."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    proto = F.normalize(prototypes, dim=-1)

    logits_a = z_a @ proto.t()
    logits_b = z_b @ proto.t()

    with torch.no_grad():
        q_a = sinkhorn_knopp(logits_a, n_iters=sinkhorn_iters, epsilon=sinkhorn_eps)
        q_b = sinkhorn_knopp(logits_b, n_iters=sinkhorn_iters, epsilon=sinkhorn_eps)

    p_a = F.log_softmax(logits_a / temperature, dim=-1)
    p_b = F.log_softmax(logits_b / temperature, dim=-1)
    loss_ab = -(q_b * p_a).sum(dim=-1).mean()
    loss_ba = -(q_a * p_b).sum(dim=-1).mean()
    return 0.5 * (loss_ab + loss_ba)


def sigreg_loss(
    x: torch.Tensor,
    global_step: int,
    num_slices: int = 256,
    num_points: int = 17,
) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (Epps-Pulley), LeJEPA Alg. 1.

    Pushes batch embeddings toward N(0, I) via random 1D slice normality tests.
    """
    n, _ = x.shape
    device = x.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(global_step))

    directions = torch.randn(x.size(1), num_slices, generator=gen, device=device)
    directions = directions / directions.norm(p=2, dim=0, keepdim=True)
    proj = x @ directions  # (N, num_slices)

    t = torch.linspace(-5.0, 5.0, num_points, device=device)
    target_cf = torch.exp(-0.5 * t**2)

    x_t = proj.unsqueeze(-1) * t  # (N, num_slices, T)
    ecf = torch.exp(1j * x_t).mean(dim=0)  # (num_slices, T)
    err = (ecf - target_cf).abs().square() * target_cf
    per_slice = torch.trapz(err.real, t, dim=1) * n
    return per_slice.mean()


@dataclass
class TripletJEPALoss:
    margin: float = 0.2
    triplet_weight: float = 0.5
    vicreg_inv_weight: float = 0.0
    vicreg_var_weight: float = 0.0
    vicreg_cov_weight: float = 0.0
    sigreg_weight: float = 0.0
    sigreg_num_slices: int = 256
    uniformity_weight: float = 0.0
    uniformity_t: float = 2.0
    proto_weight: float = 0.0
    proto_num: int = 100
    proto_temperature: float = 0.1
    sinkhorn_iters: int = 3
    sinkhorn_eps: float = 0.05
    triplet_cosine: bool = False
    jepa_augment_triplet_uniformity: bool = False
    jepa_infonce_augment: bool = False
    jepa_infonce_vicreg: bool = False
    jepa_infonce_sigreg: bool = False
    sigreg_inv_weight: float = 0.0
    distortion_ranking: bool = False
    rank_weight: float = 1.0
    aug_align_weight: float = 0.0
    align_weight: float = 1.0
    jepa_infonce_weight: float = 1.0
    aug_infonce_weight: float = 1.0
    infonce_temperature: float = 0.1

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
        z_sigreg: torch.Tensor | None = None,
        z_uniformity: torch.Tensor | None = None,
        z_enc_aug: torch.Tensor | None = None,
        prototypes: torch.Tensor | None = None,
        z_proto_a: torch.Tensor | None = None,
        z_proto_b: torch.Tensor | None = None,
        z_rank_levels: torch.Tensor | None = None,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        stats: dict[str, float] = {}
        if self.distortion_ranking:
            if z_rank_levels is None:
                raise ValueError("z_rank_levels is required for distortion_ranking")
            l_rank, rank_stats = ranking_cosine_loss(z_anchor, z_rank_levels, margin=self.margin)
            total = self.rank_weight * l_rank
            stats.update(rank_stats)
            if z_inv_a is not None and z_inv_b is not None:
                stats["align"] = jepa_cosine_loss(z_inv_a, z_inv_b).item()
                if self.jepa_infonce_weight > 0:
                    l_jepa_nce = infonce_loss(z_inv_a, z_inv_b, self.infonce_temperature)
                    total = total + self.jepa_infonce_weight * l_jepa_nce
                    stats["jepa_infonce"] = l_jepa_nce.item()
                if self.align_weight > 0:
                    total = total + self.align_weight * jepa_cosine_loss(z_inv_a, z_inv_b)
            if z_enc_aug is not None and z_inv_b is not None:
                stats["aug_align"] = jepa_cosine_loss(z_enc_aug, z_inv_b).item()
                if self.aug_infonce_weight > 0:
                    l_aug_nce = infonce_loss(z_enc_aug, z_inv_b, self.infonce_temperature)
                    total = total + self.aug_infonce_weight * l_aug_nce
                    stats["aug_infonce"] = l_aug_nce.item()
                if self.aug_align_weight > 0:
                    total = total + self.aug_align_weight * jepa_cosine_loss(z_enc_aug, z_inv_b)
            stats["loss"] = total.item()
            return total, stats

        l_jepa = jepa_cosine_loss(z_anchor, z_positive)
        stats = {"jepa": l_jepa.item()}
        use_vicreg = (
            self.vicreg_inv_weight > 0
            or self.vicreg_var_weight > 0
            or self.vicreg_cov_weight > 0
        )
        use_sigreg = self.sigreg_weight > 0
        use_uniformity = self.uniformity_weight > 0
        use_proto = self.proto_weight > 0

        if use_sigreg and not self.jepa_infonce_sigreg:
            if z_inv_a is None or z_inv_b is None or z_sigreg is None:
                raise ValueError("z_inv_a, z_inv_b, and z_sigreg are required for SIGReg")
            l_inv = vicreg_invariance(z_inv_a, z_inv_b)
            l_sig = sigreg_loss(z_sigreg, global_step, self.sigreg_num_slices)
            lam = self.sigreg_weight
            total = (1.0 - lam) * l_inv + lam * l_sig
            stats["sigreg_inv"] = l_inv.item()
            stats["sigreg"] = l_sig.item()
        elif self.jepa_infonce_sigreg:
            if z_inv_a is None or z_inv_b is None:
                raise ValueError("z_inv_a and z_inv_b are required for jepa_infonce_sigreg")
            if z_vicreg_a is None or z_vicreg_b is None or z_sigreg is None:
                raise ValueError(
                    "z_vicreg_a, z_vicreg_b, and z_sigreg are required for jepa_infonce_sigreg"
                )
            total = torch.tensor(0.0, device=z_inv_a.device)
            stats["align"] = jepa_cosine_loss(z_inv_a, z_inv_b).item()
            if z_enc_aug is not None:
                stats["aug_align"] = jepa_cosine_loss(z_enc_aug, z_inv_b).item()
            if self.jepa_infonce_weight > 0:
                l_jepa_nce = infonce_loss(z_inv_a, z_inv_b, self.infonce_temperature)
                total = total + self.jepa_infonce_weight * l_jepa_nce
                stats["jepa_infonce"] = l_jepa_nce.item()
            if self.sigreg_inv_weight > 0:
                l_inv = vicreg_invariance(z_vicreg_a, z_vicreg_b)
                total = total + self.sigreg_inv_weight * l_inv
                stats["sigreg_inv"] = l_inv.item()
            if self.sigreg_weight > 0:
                l_sig = sigreg_loss(z_sigreg, global_step, self.sigreg_num_slices)
                total = total + self.sigreg_weight * l_sig
                stats["sigreg"] = l_sig.item()
            if self.align_weight > 0:
                total = total + self.align_weight * jepa_cosine_loss(z_inv_a, z_inv_b)
            if self.aug_align_weight > 0 and z_enc_aug is not None:
                total = total + self.aug_align_weight * jepa_cosine_loss(z_enc_aug, z_inv_b)
            if use_uniformity:
                if z_uniformity is None:
                    raise ValueError("z_uniformity is required when uniformity_weight > 0")
                l_unif = uniformity_loss(z_uniformity, self.uniformity_t)
                total = total + self.uniformity_weight * l_unif
                stats["uniformity"] = l_unif.item()
        elif self.jepa_infonce_vicreg:
            if z_inv_a is None or z_inv_b is None:
                raise ValueError("z_inv_a and z_inv_b are required for jepa_infonce_vicreg")
            if z_vicreg_a is None or z_vicreg_b is None:
                raise ValueError("z_vicreg_a and z_vicreg_b are required for jepa_infonce_vicreg")
            total = torch.tensor(0.0, device=z_inv_a.device)
            stats["align"] = jepa_cosine_loss(z_inv_a, z_inv_b).item()
            if z_enc_aug is not None:
                stats["aug_align"] = jepa_cosine_loss(z_enc_aug, z_inv_b).item()
            if self.jepa_infonce_weight > 0:
                l_jepa_nce = infonce_loss(z_inv_a, z_inv_b, self.infonce_temperature)
                total = total + self.jepa_infonce_weight * l_jepa_nce
                stats["jepa_infonce"] = l_jepa_nce.item()
            if self.vicreg_inv_weight > 0:
                l_inv = vicreg_invariance(z_vicreg_a, z_vicreg_b)
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
            if self.align_weight > 0:
                total = total + self.align_weight * jepa_cosine_loss(z_inv_a, z_inv_b)
            if self.aug_align_weight > 0 and z_enc_aug is not None:
                total = total + self.aug_align_weight * jepa_cosine_loss(z_enc_aug, z_inv_b)
            if use_uniformity:
                if z_uniformity is None:
                    raise ValueError("z_uniformity is required when uniformity_weight > 0")
                l_unif = uniformity_loss(z_uniformity, self.uniformity_t)
                total = total + self.uniformity_weight * l_unif
                stats["uniformity"] = l_unif.item()
        elif self.jepa_infonce_augment:
            if z_inv_a is None or z_inv_b is None:
                raise ValueError("z_inv_a and z_inv_b are required for jepa_infonce_augment")
            total = torch.tensor(0.0, device=z_inv_a.device)
            stats["align"] = jepa_cosine_loss(z_inv_a, z_inv_b).item()
            if z_enc_aug is not None:
                stats["aug_align"] = jepa_cosine_loss(z_enc_aug, z_inv_b).item()
            if self.jepa_infonce_weight > 0:
                l_jepa_nce = infonce_loss(z_inv_a, z_inv_b, self.infonce_temperature)
                total = total + self.jepa_infonce_weight * l_jepa_nce
                stats["jepa_infonce"] = l_jepa_nce.item()
            if self.aug_infonce_weight > 0:
                if z_enc_aug is None:
                    raise ValueError("z_enc_aug is required when aug_infonce_weight > 0")
                l_aug_nce = infonce_loss(z_enc_aug, z_inv_b, self.infonce_temperature)
                total = total + self.aug_infonce_weight * l_aug_nce
                stats["aug_infonce"] = l_aug_nce.item()
            if self.align_weight > 0:
                total = total + self.align_weight * jepa_cosine_loss(z_inv_a, z_inv_b)
            if self.aug_align_weight > 0 and z_enc_aug is not None:
                total = total + self.aug_align_weight * jepa_cosine_loss(z_enc_aug, z_inv_b)
            if use_uniformity:
                if z_uniformity is None:
                    raise ValueError("z_uniformity is required when uniformity_weight > 0")
                l_unif = uniformity_loss(z_uniformity, self.uniformity_t)
                total = total + self.uniformity_weight * l_unif
                stats["uniformity"] = l_unif.item()
        elif self.jepa_augment_triplet_uniformity:
            if z_inv_a is None or z_inv_b is None:
                raise ValueError("z_inv_a and z_inv_b are required for jepa_augment_uniformity")
            l_align = jepa_cosine_loss(z_inv_a, z_inv_b)
            total = torch.tensor(0.0, device=z_inv_a.device)
            stats["align"] = l_align.item()
            if self.align_weight > 0:
                total = total + self.align_weight * l_align
            if z_enc_aug is not None:
                l_aug_align = jepa_cosine_loss(z_enc_aug, z_inv_b)
                stats["aug_align"] = l_aug_align.item()
                if self.aug_align_weight > 0:
                    total = total + self.aug_align_weight * l_aug_align
            if use_uniformity:
                if z_uniformity is None:
                    raise ValueError("z_uniformity is required when uniformity_weight > 0")
                l_unif = uniformity_loss(z_uniformity, self.uniformity_t)
                total = total + self.uniformity_weight * l_unif
                stats["uniformity"] = l_unif.item()
            if self.triplet_weight > 0:
                if z_enc_aug is None or z_negative is None:
                    raise ValueError(
                        "z_enc_aug and z_negative are required when triplet_weight > 0"
                    )
                l_triplet = triplet_cosine_loss(
                    z_enc_aug, z_inv_b, z_negative, margin=self.margin
                )
                total = total + self.triplet_weight * l_triplet
                stats["triplet"] = l_triplet.item()
        elif self.triplet_cosine:
            if z_inv_a is None or z_inv_b is None:
                raise ValueError("z_inv_a and z_inv_b are required for triplet_uniformity")
            l_align = jepa_cosine_loss(z_inv_a, z_inv_b)
            total = l_align
            stats["align"] = l_align.item()
            if use_uniformity:
                if z_uniformity is None:
                    raise ValueError("z_uniformity is required when uniformity_weight > 0")
                l_unif = uniformity_loss(z_uniformity, self.uniformity_t)
                total = total + self.uniformity_weight * l_unif
                stats["uniformity"] = l_unif.item()
            if self.triplet_weight > 0:
                if z_negative is None:
                    raise ValueError("z_negative is required when triplet_weight > 0")
                l_triplet = triplet_cosine_loss(
                    z_inv_a, z_inv_b, z_negative, margin=self.margin
                )
                total = total + self.triplet_weight * l_triplet
                stats["triplet"] = l_triplet.item()
        elif use_uniformity or use_proto:
            if z_inv_a is None or z_inv_b is None:
                raise ValueError("z_inv_a and z_inv_b are required for sphere alignment")
            l_align = jepa_cosine_loss(z_inv_a, z_inv_b)
            total = l_align
            stats["align"] = l_align.item()
            if use_uniformity:
                if z_uniformity is None:
                    raise ValueError("z_uniformity is required when uniformity_weight > 0")
                l_unif = uniformity_loss(z_uniformity, self.uniformity_t)
                total = total + self.uniformity_weight * l_unif
                stats["uniformity"] = l_unif.item()
            if use_proto:
                if prototypes is None or z_proto_a is None or z_proto_b is None:
                    raise ValueError(
                        "prototypes, z_proto_a, and z_proto_b are required when proto_weight > 0"
                    )
                l_swav = swav_loss(
                    z_proto_a,
                    z_proto_b,
                    prototypes,
                    temperature=self.proto_temperature,
                    sinkhorn_iters=self.sinkhorn_iters,
                    sinkhorn_eps=self.sinkhorn_eps,
                )
                total = total + self.proto_weight * l_swav
                stats["swav"] = l_swav.item()
        elif use_vicreg:
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

        if (
            self.triplet_weight > 0
            and not self.triplet_cosine
            and not self.jepa_augment_triplet_uniformity
            and not self.jepa_infonce_augment
            and not self.jepa_infonce_vicreg
            and not self.jepa_infonce_sigreg
        ):
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
