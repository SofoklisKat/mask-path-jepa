# Experiment Plan — Triplet-JEPA

**Status:** Planned (not yet executed)  
**Paper:** Challenges Paper 2026  
**Goal:** Low-resource, reproducible evidence that triplet margin loss improves JEPA encoder geometry.

This document lists **every run** we intend to execute, in order, with fixed hyperparameters, commands, and expected outputs. See [EXPERIMENTS.md](EXPERIMENTS.md) for the scientific protocol and [ssl_jepa_triplet/NOTES.md](ssl_jepa_triplet/NOTES.md) for literature context.

---

## 0. Shared settings (all runs unless noted)

| Setting | Value |
|---------|-------|
| Dataset (primary) | CIFAR-10, 50k train / 10k test |
| Backbone | SmallResNet encoder + MLP predictor (~600K trainable params) |
| Embed dim | 256 |
| EMA momentum | 0.996 |
| Masking | Multi-block, `mask_ratio=0.6`, 8×8 grid of 8×8 blocks on 32×32 |
| Augmentation (train) | RandomCrop(32, pad=4) + RandomHorizontalFlip |
| Optimizer | AdamW, `lr=3e-4`, `weight_decay=1e-4`, cosine schedule |
| Batch size | 128 (64 for smoke only) |
| Seed | 42 |
| Eval metrics | k-NN@20 (cosine), linear probe (50 SGD epochs on frozen encoder) |
| Inference | Encoder only; predictor discarded at eval |

**Loss (when λ > 0):**

```
L = L_JEPA + λ · L_triplet
L_JEPA     = mean(1 - cos(ẑ, z+))
L_triplet  = mean(relu(||ẑ-z+||² - ||ẑ-z-||² + m))
```

Default margin `m = 0.2`.

---

## 1. Execution phases

```
Phase A  Pipeline validation     (1 run,  ~5 min CPU)
Phase B  Main comparison        (3 runs, ~1–2 h GPU / ~4–6 h CPU)
Phase C  λ ablation             (3 runs, same budget as Phase B)
Phase D  Margin ablation        (2 runs, same budget as Phase B)
Phase E  Negative-mode check    (1 run,  subset of Phase B)
```

**Total planned runs: 10**  
**Primary result table:** Phase B (main comparison). Phases C–E support ablation figures.

---

## 2. Phase A — Pipeline validation

Sanity-check that training, eval, and result logging work end-to-end. **No paper numbers from this phase.**

| ID | Run name | Purpose |
|----|----------|---------|
| A1 | `quick_smoke` | Verify code path before long runs |

### A1 — `quick_smoke`

| Hyperparameter | Value |
|----------------|-------|
| `dataset` | cifar10 |
| `train_subset` | 2,000 |
| `epochs` | 5 |
| `batch_size` | 64 |
| `triplet_weight` (λ) | 0.5 |
| `negative_mode` | class |
| `margin` | 0.2 (default) |
| `eval_every` | 5 |
| `probe_epochs` | 10 |

**Command:**
```bash
PYTHONPATH=. python train.py --config configs/quick_smoke.json
```

**Output:** `results/quick_smoke/`  
**Pass criteria:** completes without error; `results.json` contains `knn` and `linear_probe` keys.

---

## 3. Phase B — Main comparison (paper Table 1)

Three methods, **identical compute**, differing only in triplet term and negative mining. This is the **core paper result**.

| ID | Run name | λ | Negative mode | Role |
|----|----------|---|---------------|------|
| B1 | `jepa_baseline` | 0.0 | instance | Pure JEPA regression (control) |
| B2 | `triplet_instance` | 0.5 | instance | Unsupervised triplet regularization |
| B3 | `triplet_class` | 0.5 | class | Class-aware negative mining |

### Shared hyperparameters (B1–B3)

| Hyperparameter | Value |
|----------------|-------|
| `dataset` | cifar10 |
| `epochs` | 100 |
| `batch_size` | 128 |
| `margin` | 0.2 |
| `mask_ratio` | 0.6 |
| `eval_every` | 10 |
| `knn_k` | 20 |
| `probe_epochs` | 50 |
| `seed` | 42 |

### B1 — `jepa_baseline`

| Variable | Value |
|----------|-------|
| `triplet_weight` | **0.0** |
| `negative_mode` | instance (unused when λ=0) |

**Command:**
```bash
PYTHONPATH=. python train.py --config configs/cifar10_jepa.json
```

**Output:** `results/jepa_baseline/`

---

### B2 — `triplet_instance`

| Variable | Value |
|----------|-------|
| `triplet_weight` | **0.5** |
| `negative_mode` | **instance** (batch shuffle) |

**Command:**
```bash
PYTHONPATH=. python train.py \
  --config configs/cifar10_jepa.json \
  --run-name triplet_instance \
  --triplet-weight 0.5 \
  --negative-mode instance
```

**Output:** `results/triplet_instance/`

---

### B3 — `triplet_class`

| Variable | Value |
|----------|-------|
| `triplet_weight` | **0.5** |
| `negative_mode` | **class** |

**Command:**
```bash
PYTHONPATH=. python train.py --config configs/cifar10_triplet_jepa.json
# equivalent: --run-name triplet_class (config uses run_name triplet_jepa; we rename for clarity)
```

*Note: config file `cifar10_triplet_jepa.json` uses `run_name: triplet_jepa`. For the paper table we will rename the output folder to `triplet_class` for consistency with B2.*

**Output:** `results/triplet_class/` (or `results/triplet_jepa/` — normalize before writing table)

---

### Batch command (all of Phase B)

```bash
PYTHONPATH=. python scripts/run_paper_experiments.py \
  --dataset cifar10 \
  --epochs 100 \
  --output-dir ./results/paper_cifar10
```

**Aggregated output:**
- `results/paper_cifar10/TABLE.md`
- `results/paper_cifar10/summary.json`

### Expected paper table (to fill after runs)

| Method | λ | Negatives | k-NN@20 ↑ | Linear probe ↑ | Params |
|--------|---|-----------|-----------|----------------|--------|
| JEPA baseline (B1) | 0 | — | TBD | TBD | 0.6M |
| Triplet-JEPA instance (B2) | 0.5 | instance | TBD | TBD | 0.6M |
| Triplet-JEPA class (B3) | 0.5 | class | TBD | TBD | 0.6M |

**Primary hypothesis:** B2 or B3 beats B1 on **k-NN**; linear probe may be equal or slightly better.

---

## 4. Phase C — λ ablation (paper Figure 1)

Fix `negative_mode=class`, sweep triplet weight. Isolates how much margin regularization helps.

| ID | Run name | λ | Other settings |
|----|----------|---|----------------|
| C1 | `ablation_lambda_0.1` | 0.1 | same as B3 |
| C2 | `ablation_lambda_0.5` | 0.5 | same as B3 (= B3, skip if B3 already run) |
| C3 | `ablation_lambda_1.0` | 1.0 | same as B3 |

**Base config:** `configs/cifar10_triplet_jepa.json`

**Commands:**
```bash
for LAMBDA in 0.1 0.5 1.0; do
  PYTHONPATH=. python train.py \
    --config configs/cifar10_triplet_jepa.json \
    --run-name "ablation_lambda_${LAMBDA}" \
    --triplet-weight "$LAMBDA"
done
```

**Output:** `results/ablation_lambda_*/`

**Figure:** plot λ (x) vs k-NN and linear probe (y). Expect λ too large may hurt JEPA regression (higher `jepa` loss).

| λ | k-NN@20 | Linear probe |
|---|---------|--------------|
| 0.0 | (from B1) | (from B1) |
| 0.1 | TBD | TBD |
| 0.5 | TBD | TBD |
| 1.0 | TBD | TBD |

---

## 5. Phase D — Margin ablation

Fix `λ=0.5`, `negative_mode=class`, sweep FaceNet margin `m`.

| ID | Run name | Margin m |
|----|----------|----------|
| D1 | `ablation_margin_0.1` | 0.1 |
| D2 | `ablation_margin_0.2` | 0.2 (= default, same as B3; skip if B3 run) |

**Commands:**
```bash
for M in 0.1 0.2; do
  PYTHONPATH=. python train.py \
    --config configs/cifar10_triplet_jepa.json \
    --run-name "ablation_margin_${M}"
done
```

*Margin is not yet a CLI flag; add `--margin` to `train.py` or duplicate JSON configs before running Phase D.*

**Planned config files (to create before run):**
- `configs/ablation_margin_0.1.json` — copy `cifar10_triplet_jepa.json`, set `"margin": 0.1`, `"run_name": "ablation_margin_0.1"`
- `configs/ablation_margin_0.2.json` — `"margin": 0.2` (reference only)

**Output:** `results/ablation_margin_*/`

| Margin m | k-NN@20 | Linear probe |
|----------|---------|--------------|
| 0.1 | TBD | TBD |
| 0.2 | TBD | TBD |

---

## 6. Phase E — Negative-mode comparison (supplementary)

Direct instance vs class at λ=0.5. Overlaps with B2 vs B3; included here for explicit ablation narrative.

| ID | Run name | Negative mode | Notes |
|----|----------|---------------|-------|
| E1 | `triplet_instance` | instance | Same as B2 |
| E2 | `triplet_class` | class | Same as B3 |

**No extra runs if Phase B is complete.** Report as a paired comparison in text:

> "Class-aware negatives outperform batch shuffle by X% k-NN at equal λ."

---

## 7. Quick-dev subset (optional, not for paper)

For fast iteration during development. **Do not cite these numbers.**

| ID | Run name | Subset | Epochs | Command |
|----|----------|--------|--------|---------|
| Q1 | `jepa_baseline` | 5k | 20 | `run_paper_experiments.py --quick` |
| Q2 | `triplet_instance` | 5k | 20 | same |
| Q3 | `triplet_class` | 5k | 20 | same |

```bash
PYTHONPATH=. python scripts/run_paper_experiments.py --quick
```

**Output:** `results/paper_cifar10/` (overwrites full-run table if same dir — use separate `--output-dir ./results/quick_dev` for safety)

---

## 8. Per-run artifacts

Every run produces:

```
results/<run_name>/
  config.json      # frozen hyperparameters
  results.json     # full metric history + final epoch
  encoder.pt       # encoder weights for downstream use
```

`results.json` final epoch fields of interest:

| Field | Description |
|-------|-------------|
| `loss` | Total training loss |
| `jepa` | Cosine JEPA term |
| `triplet` | Margin term (0 when λ=0) |
| `knn` | k-NN@20 accuracy |
| `linear_probe` | Linear probe accuracy |

---

## 9. Compute budget estimate

| Phase | Runs | Epochs | Data | Est. time (1× GPU) | Est. time (CPU) |
|-------|------|--------|------|--------------------|-----------------|
| A | 1 | 5 | 2k | < 2 min | ~5 min |
| B | 3 | 100 | 50k | ~30–45 min each | ~2 h each |
| C | 2–3 | 100 | 50k | ~30–45 min each | ~2 h each |
| D | 1–2 | 100 | 50k | ~30–45 min each | ~2 h each |

**Full paper suite (A + B + C + D, skipping duplicates):** ~8–12 GPU-hours or ~24–36 CPU-hours.

**Minimum publishable set:** Phase A + Phase B only (~1.5–2 GPU-hours).

---

## 10. Run order (checklist)

Execute in this order. Do not start Phase B until Phase A passes.

- [ ] **A1** `quick_smoke` — pipeline validation
- [ ] **B1** `jepa_baseline` — main table control
- [ ] **B2** `triplet_instance` — main table unsupervised
- [ ] **B3** `triplet_class` — main table class-aware
- [ ] Aggregate → `results/paper_cifar10/TABLE.md`
- [ ] **C1** `ablation_lambda_0.1`
- [ ] **C3** `ablation_lambda_1.0` (C2 = B3, skip)
- [ ] **D1** `ablation_margin_0.1` (after adding config/CLI)
- [ ] Write paper Table 1 + Figure 1 from results

---

## 11. Out of scope (not planned)

These are explicitly **not** in the current experiment plan:

| Item | Reason |
|------|--------|
| ImageNet / full Tiny ImageNet | Beyond low-resource budget |
| ViT / patch-level JEPA | Future work (see NOTES §10) |
| EMA on/off ablation | CLI toggle not implemented yet |
| Video (V-JEPA) | Different modality |
| Proxy-NCA | Optional extension, not baseline |
| MNIST main results | CIFAR-10 only for paper credibility |
| Multi-seed (3×) | Add if reviewers request; seed=42 for now |
| Fine-tuning eval | Linear probe + k-NN only |

---

## 12. Success criteria

| Criterion | Target |
|-----------|--------|
| Pipeline | Phase A completes, metrics logged |
| Main result | B2 or B3 k-NN > B1 k-NN by ≥ 1–2 pts absolute |
| JEPA not broken | B2/B3 `jepa` loss within ~20% of B1 at convergence |
| Reproducibility | All runs have `config.json` + `results.json` |
| Paper table | Phase B fills Table 1 with three rows |

Failure modes to report honestly:
- Triplet helps k-NN but hurts linear probe → geometry vs separability trade-off
- No gain at λ=0.5 → report λ sweep (Phase C) for sweet spot
- Class negatives ≈ instance → labels add little at this scale

---

## 13. Config file index

| File | Maps to run |
|------|-------------|
| `configs/quick_smoke.json` | A1 |
| `configs/cifar10_jepa.json` | B1 |
| `configs/cifar10_triplet_jepa.json` | B3 (rename run to `triplet_class`) |
| `configs/ablation_margin_0.1.json` | D1 (to create) |
| `configs/ablation_lambda_*.json` | C1–C3 (optional; CLI overrides suffice) |

---

*Plan version 1.0 — created before any experiment execution.*
