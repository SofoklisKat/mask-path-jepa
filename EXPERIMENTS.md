# Experimental Protocol — Triplet-JEPA (Low-Resource)

Reproducible benchmark for **Challenges Paper 2026**. Designed to run on a single consumer GPU or CPU in under a few hours.

---

## 1. Research question

> Does a **triplet margin term** on predictor outputs improve representation geometry of a **JEPA** model, without replacing the core latent-prediction objective?

**Hypothesis:** JEPA regression aligns corrupt→clean latents; triplet enforces inter-instance (or inter-class) margins → better k-NN / linear probe at equal compute.

---

## 2. Method (paper-ready summary)

### Architecture

| Component | Role | Trainable | Inference |
|-----------|------|-----------|-----------|
| Online encoder `f_θ` | Maps image → latent | Yes | **Yes** |
| Predictor `g_φ` | Maps corrupt latent → predicted clean latent | Yes | No |
| Target encoder `f_ξ` | EMA copy of `f_θ` | No (EMA) | No |

### Views

- **Clean:** standard train augmentation (CIFAR-10: crop + flip).
- **Corrupt:** multi-block masking (`mask_ratio=0.6`, 4×4 blocks on 32×32), I-JEPA inspired.

### Loss

```
ẑ  = g_φ(f_θ(x_corrupt))          # anchor
z+ = sg(f_ξ(x_clean))             # positive (stop-gradient)
z- = sg(f_ξ(x_neg))               # negative

L_JEPA     = E[1 - cos(ẑ, z+)]
L_triplet  = E[relu(||ẑ-z+||² - ||ẑ-z-||² + m)]
L          = L_JEPA + λ L_triplet
```

- **λ = 0:** JEPA baseline.
- **λ = 0.5:** Triplet-JEPA (default).
- **m = 0.2** (FaceNet-style).

### Negative sampling

| Mode | Negative | Supervision |
|------|----------|-------------|
| `instance` | Random other image in batch | Unsupervised |
| `class` | Different class in batch | Label used only for mining |

---

## 3. Low-resource budget

| Resource | Setting | Rationale |
|----------|---------|-----------|
| Dataset | CIFAR-10 (50k train) | Standard SSL probe benchmark |
| Backbone | SmallResNet (~400K encoder params) | Fits laptop / free Colab |
| Embed dim | 256 | Matches small-scale papers |
| Epochs | 100 (full) / 20 (quick) | ~30–90 min GPU for full suite |
| Batch | 128 | Stable triplet mining |
| Optimizer | AdamW, lr=3e-4, cosine | Simple, no heavy tuning |
| EMA | m=0.996 | I-JEPA style target encoder |

**Total trainable params:** ~600K (encoder + predictor).

---

## 4. Evaluation (DINO-style)

All metrics use **frozen encoder** `f_θ` only (predictor discarded).

1. **k-NN classification** — 20-NN on L2-normalized features, cosine distance.
2. **Linear probe** — logistic regression head (SGD, 50 epochs) on train features, test on test set.

Report both; k-NN stresses geometry (triplet's intended effect), linear probe stresses semantic separability.

---

## 5. Baselines & ablations

| Run | λ | Negatives | Purpose |
|-----|---|-----------|---------|
| `jepa_baseline` | 0.0 | — | Pure JEPA regression |
| `triplet_instance` | 0.5 | batch shuffle | Unsupervised triplet |
| `triplet_class` | 0.5 | different class | Semi-supervised mining |

**Suggested ablations for paper:** λ ∈ {0.1, 0.5, 1.0}, margin m ∈ {0.1, 0.2}, with/without EMA.

---

## 6. Reproduction commands

```bash
pip install -r requirements.txt

# Quick smoke (~5 min CPU, validates pipeline)
python train.py --config configs/quick_smoke.json

# Single full run
python train.py --config configs/cifar10_triplet_jepa.json

# Full comparison table (3 baselines)
python scripts/run_paper_experiments.py --dataset cifar10 --epochs 100

# Fast comparison for iteration
python scripts/run_paper_experiments.py --quick
```

Results land in `results/<run_name>/`:
- `config.json` — full hyperparameters
- `results.json` — metrics history + final numbers
- `encoder.pt` — frozen encoder weights

Aggregate table: `results/paper_cifar10/TABLE.md`

---

## 7. Reporting template (for paper)

**Table 1 — CIFAR-10 representation quality (encoder frozen)**

| Method | k-NN@20 (↑) | Linear probe (↑) | Params |
|--------|-------------|------------------|--------|
| JEPA (λ=0) | — | — | 0.6M |
| Triplet-JEPA (instance) | — | — | 0.6M |
| Triplet-JEPA (class) | — | — | 0.6M |

*Fill from `TABLE.md` after running experiments.*

**Figure 1 (optional):** λ vs k-NN curve.

---

## 8. Limitations (honest scope for low-resource paper)

- CIFAR-10 only at 32×32; no ImageNet-scale claims.
- Small CNN, not ViT — patch-level JEPA left for future work.
- Class negatives use labels at train time (semi-supervised); instance mode is fully unsupervised.
- No EMA ablation toggle in CLI yet (always on).

---

## 9. Claims you can make

- Novel **combination** (JEPA + triplet margin) with clear motivation.
- Reproducible **low-compute** protocol with public code.
- Empirical comparison: regression-only vs margin-regularized latent prediction.
- k-NN gains as evidence for **metric geometry** improvement.

## 10. Claims to avoid

- SOTA on CIFAR-10 (SimCLR / DINO / I-JEPA use larger models).
- "First JEPA implementation" — cite I-JEPA / V-JEPA.
- Video / robotics without additional experiments.

---

*Protocol version 1.0 — TripletJEPA repo.*
