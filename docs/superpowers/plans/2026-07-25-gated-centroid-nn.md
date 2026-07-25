# Gated Centroid kNN + JEPA (CIFAR-10)

> **For agentic workers:** Implement task-by-task. Checkboxes track progress.

**Goal:** Add JEPA + soft/gated in-batch neighbor loss on per-image aug centroids; train CIFAR-10 first.

**Architecture:** Each batch image gets M augs → mean embedding (centroid). Pairwise cosine on B centroids; keep top-1 if sim ≥ τ (else no pair). Neighbor loss = weighted mean cosine distance on gated pairs. JEPA mask loss always on. τ can anneal over epochs.

**Tech Stack:** PyTorch, existing TripletJEPA trainer.

---

## File map

- `tripletjepa/losses.py` — `gated_centroid_nn_loss`
- `tripletjepa/views.py` — multi-aug helper for centroids
- `tripletjepa/train.py` — mode `jepa_gated_centroid_nn`, config fields, train step
- `train.py` — CLI choice
- `configs/cifar10_jepa_gated_centroid_nn.json` — CIFAR-10 run

## Tasks

- [ ] Loss + view helpers
- [ ] Wire train loop + resolve_training_mode
- [ ] Config + smoke one-batch forward
