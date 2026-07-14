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
| Predictor `g_φ` | Maps corrupt latent → predicted clean latent | Yes (if `anchor_mode=predictor_corrupt`) | No |
| Target encoder `f_ξ` | EMA copy of `f_θ` (optional) | No (EMA) | No |

Set `use_ema_target: false` for a **single-encoder** setup: positives/negatives use
`encoder(clean/scramble).detach()` with stop-gradient instead of an EMA teacher.
Use `anchor_mode: encoder_corrupt` to align JEPA/triplet on encoder features.

### Views

- **Clean:** standard train augmentation (CIFAR-10: crop + flip).
- **Corrupt (default):** multi-block masking (`mask_ratio=0.6`, 4×4 blocks on 32×32), I-JEPA inspired.
- **Corrupt (curriculum):** `corrupt_schedule: blur_to_mask` or `mask_curriculum` — corruption is **patch-local** on a `patch_size×patch_size` grid (default 4×4 on 32×32, 64 tiles total). A random subset of tiles is blurred per image; unselected tiles stay sharp. Fraction ramps `patch_blur_ratio_min → mask_ratio` (default **10% → 60%**), σ per blurred tile ramps `blur_sigma_min → blur_sigma_max`. No whole-image blur, no zero masking.

### Loss

```
ẑ  = g_φ(f_θ(x_corrupt))          # anchor
z+ = sg(f_ξ(x_clean))             # positive (stop-gradient)
z- = sg(f_ξ(x_neg))               # negative

L_JEPA     = E[1 - cos(ẑ, z+)]
L_triplet  = E[relu(||â-ṕ||² - ||â-ñ||² + m)]   # â,ṕ,ñ = L2-normalized embeddings
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
| `scramble` | Same image, shuffled patch grid | Unsupervised |
| `scramble_class` | Scramble + different-class batch embedding | Layout + semantics |

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
| `triplet_scramble` | 0.5 | patch shuffle (same image) | Layout-breaking negative |

Optional flags: `use_ema_target` (default `true`), `anchor_mode` (`predictor_corrupt` | `encoder_corrupt`).

### `latent_triplet` mode (single encoder, no EMA)

High-level training mode that replaces the EMA teacher with a latent-space triplet regularizer:

```text
training_mode: latent_triplet

anchor   = fθ(x_corrupt)              # grad
positive = stopgrad(fθ(x_clean))      # JEPA invariance
negative = stopgrad(fθ(x_scramble))   # triplet: broken global layout
negative = stopgrad(fθ(x_class))     # triplet: different-class context

L_triplet = 0.5 · [margin(anchor, pos, scramble) + margin(anchor, pos, class)]

L = L_JEPA + λ L_triplet
```

- No second encoder, no EMA, no predictor in the training path.
- Triplet acts as the geometric regularizer (scramble = wrong layout).
- Eval still uses frozen `fθ` on clean images.

Configs: `configs/cifar100_latent_triplet_jepa.json` (λ=0), `configs/cifar100_latent_triplet.json` (λ=0.05).

### `latent_vicreg` mode (single encoder, no EMA)

Replace EMA/triplet with VICReg anti-collapse terms on encoder outputs:

```text
training_mode: latent_vicreg

# Alignment head (predictor) — recommended
ẑ = gφ(fθ(x_corrupt))                  # JEPA anchor
z+ = stopgrad(fθ(x_clean))             # JEPA positive
anchor_mode: predictor_corrupt

# Direct encoder path (ablation)
ẑ = fθ(x_corrupt)
anchor_mode: encoder_corrupt

# VICReg on encoder outputs (eval space); variance + covariance on both views
L = λ·MSE(z_inv_a, z_inv_b) + μ·L_var(fθ(x_corrupt), fθ(x_clean)) + β·L_cov(...)
```

- **Invariance** `λ=25`: MSE between corrupt/clean representations (replaces JEPA cosine in the loss).
- **Align mode:** `z_inv_a = gφ(fθ(x_corrupt))`, `z_inv_b = stopgrad(fθ(x_clean))`.
- **Encoder mode:** `z_inv_a = fθ(x_corrupt)`, `z_inv_b = fθ(x_clean)` (both branches train).
- JEPA cosine is still **logged** as `jepa(log)` for monitoring, not optimized.

Configs:
- `configs/cifar100_latent_vicreg_align.json` — predictor + VICReg (recommended)
- `configs/cifar100_latent_vicreg.json` — encoder-only + VICReg (ablation)

**Suggested ablations for paper:** λ ∈ {0.05, 0.1, 0.5}, margin m ∈ {0.1, 0.2}, with/without EMA.

**Heavy triplet (λ=25, mirrors VICReg α=β=25):** `latent_triplet_heavy`, `latent_triplet_align_heavy`, `triplet_heavy_ema` — expect training instability / collapsed eval (λ=0.5 already hurt badly).

```bash
PYTHONPATH=. python scripts/run_paper_experiments.py \
  --dataset cifar100 --suite ablation --only latent_triplet_heavy
```

---

## 6. Reproduction commands

```bash
pip install -r requirements.txt

# Quick smoke (~5 min CPU, validates pipeline)
python train.py --config configs/quick_smoke.json

# Single full run
python train.py --config configs/cifar10_triplet_jepa.json

# Full comparison table (3 baselines + optional scramble)
python scripts/run_paper_experiments.py --dataset cifar10 --epochs 100

# CIFAR-100 main suite (all experiments)
PYTHONPATH=. python scripts/run_paper_experiments.py \
  --dataset cifar100 --suite full --output-dir ./results/paper_cifar100

# Skip runs already finished; only aggregate table
PYTHONPATH=. python scripts/run_paper_experiments.py \
  --dataset cifar100 --suite full --output-dir ./results/paper_cifar100 \
  --skip-existing --aggregate-only

# Quick smoke (20 epochs, 5k subset)
PYTHONPATH=. python scripts/run_paper_experiments.py --dataset cifar100 --suite full --quick

# Run one experiment
PYTHONPATH=. python scripts/run_paper_experiments.py \
  --dataset cifar100 --only latent_vicreg_align

# Fast comparison for iteration
python scripts/run_paper_experiments.py --quick
```

Results land in `results/<run_name>/`:
- `config.json` — full hyperparameters
- `results.json` — metrics history + final numbers
- `encoder.pt` — frozen encoder weights

Suites: `core` (3 EMA baselines), `ema`, `latent_triplet`, `latent_vicreg`, `full` (8 runs).

Aggregate table: `results/paper_cifar100/TABLE.md`

---

## 7. CIFAR-100 results (completed)

See **`RESULTS_CIFAR100.md`** for the full table. Summary:

| Method | k-NN@20 | Linear probe |
|--------|---------|--------------|
| JEPA (λ=0) | **0.175** | **0.132** |
| Triplet instance (λ=0.5) | 0.049 | 0.050 |
| Triplet class (λ=0.5) | 0.063 | 0.053 |
| Single-encoder scramble (λ=0.05) | 0.043 | 0.034 |

**Finding:** triplet margin does not improve over JEPA-only under this protocol.

---

## 8. Reporting template (for paper)

**Table 1 — CIFAR-10 representation quality (encoder frozen)**

| Method | k-NN@20 (↑) | Linear probe (↑) | Params |
|--------|-------------|------------------|--------|
| JEPA (λ=0) | — | — | 0.6M |
| Triplet-JEPA (instance) | — | — | 0.6M |
| Triplet-JEPA (class) | — | — | 0.6M |

*Fill from `TABLE.md` after running experiments.*

**Figure 1 (optional):** λ vs k-NN curve.

---

## 9. Limitations (honest scope for low-resource paper)

- CIFAR-10 / CIFAR-100 at 32×32; no ImageNet-scale claims.
- Small CNN, not ViT — patch-level JEPA left for future work.
- Class negatives use labels at train time (semi-supervised); instance mode is fully unsupervised.
- Triplet term is applied on predictor outputs by default; eval uses encoder only (space mismatch).
- EMA target is a training stabilizer, not part of the core loss definition.

---

## 10. Claims you can make

- Novel **combination** (JEPA + triplet margin) with clear motivation and negative result on CIFAR-100.
- Reproducible **low-compute** protocol with public code.
- Empirical comparison: regression-only vs margin-regularized latent prediction.
- Honest report: triplet did not improve k-NN / linear probe at λ=0.5.

## 11. Claims to avoid

- SOTA on CIFAR-10/100 (SimCLR / DINO / I-JEPA use larger models).
- "First JEPA implementation" — cite I-JEPA / V-JEPA.
- "Triplet improves geometry" — not supported by CIFAR-100 results.
- Video / robotics without additional experiments.

---

*Protocol version 1.0 — TripletJEPA repo.*
