"""Triplet-JEPA: margin-regularized joint-embedding predictive architectures."""

from tripletjepa.losses import TripletJEPALoss, jepa_cosine_loss, triplet_loss
from tripletjepa.models import TripletJEPA

__all__ = [
    "TripletJEPA",
    "TripletJEPALoss",
    "jepa_cosine_loss",
    "triplet_loss",
]
