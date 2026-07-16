# Experiment Log — TripletJEPA

Living record of runs and results. **Update this file after every experiment.**

Protocol details (architecture, loss definitions, eval): see `EXPERIMENTS.md`.

---

## Shared protocol

| Setting | Value |
|---------|--------|
| Dataset | **CIFAR-100** (main); CIFAR-10 configs exist |
| Backbone | SmallResNet (~400K encoder params) |
| Embed dim | 256 |
| Epochs | 100 |
| Batch size | 256 |
| Seed | 42 |
| Corrupt (default) | 60% block mask, 4×4 patches |
| Eval | Frozen **encoder** only — k-NN@20 + linear probe every 10 epochs |

---

## Current method (chosen direction)

```
one encoder + predictor alignment (no EMA)
+ fixed block mask
+ regularizer: uniformity / SIGReg / VICReg (under comparison)
```

| Piece | Setting |
|-------|---------|
| `training_mode` | `latent_uniformity` (current) |
| `anchor_mode` | `predictor_corrupt` |
| `use_ema_target` | `false` |
| Eval | `encoder(clean)` only |

**Dropped from main story:** EMA teacher (comparison only), triplet margin (negative result), patch-blur curriculum (no gain for SIGReg).

---

## Training modes (from code)

Defined in `tripletjepa/train.py` → `resolve_training_mode()` and `TripletJEPALoss` in `tripletjepa/losses.py`.

### Mode overview

| `training_mode` | EMA? | Main loss | Anti-collapse / geometry |
|-----------------|------|-----------|--------------------------|
| **`jepa_ema`** | yes | `L_jepa` (cosine) + optional `λ·L_triplet` | EMA teacher + optional triplet margin |
| **`latent_triplet`** | no | `L_jepa` (cosine) + optional `λ·L_triplet` | Triplet on encoder/predictor latents |
| **`latent_vicreg`** | no | `α·MSE(inv)` + `μ·L_var` + `ν·L_cov` | VICReg on encoder outputs |
| **`latent_sigreg`** | no | `(1-λ)·MSE(inv)` + `λ·SIGReg` | LeJEPA-style Gaussian regularization |
| **`latent_uniformity`** | no | `cosine(align)` + `w·Uniformity` [+ `w_p·SwAV`] | Hypersphere spread (+ optional prototypes) |
| **`latent_triplet_uniformity`** | no | `cosine(align)` + `λ·cosine_triplet` + `w·Uniformity` | Same-image scramble negative (label-free) |

All modes use **corrupt + clean views**. Eval always uses frozen **`encoder(clean)`**.

---

### 1. `jepa_ema` (original protocol)

```
anchor   = predictor(encoder(x_corrupt))
positive = stopgrad(target_encoder(x_clean))    # EMA copy of online encoder
negative = mined from batch (if λ > 0)

L = L_jepa + λ · L_triplet
```

| Axis | Options |
|------|---------|
| `anchor_mode` | `predictor_corrupt` (default) |
| `negative_mode` | `instance`, `class`, `scramble`, `scramble_class` |
| `corrupt_schedule` | `block` (default), `blur_to_mask`, `mask_curriculum` |
| `triplet_weight` | 0 = pure JEPA |

**Presets / configs:**

| Preset | λ | Negatives | Corrupt | Tested? |
|--------|---|-----------|---------|---------|
| `jepa_baseline` | 0 | — | block | ✅ k-NN 0.175 @ ep100 |
| JEPA peak (manual) | 0 | — | block | ✅ k-NN **0.191** @ ep50 |
| `triplet_instance` | 0.5 | instance | block | ✅ failed |
| `triplet_class` | 0.5 | class | block | ✅ failed |
| `triplet_scramble` | 0.05 | scramble | block | ✅ poor @ ep10 |
| `triplet_heavy_ema` | 25 | scramble_class | block | 🔲 not run |
| `jepa_blur_curriculum` | 0 | — | blur_to_mask | 🔲 not run |
| `cifar100_jepa.json` | 0 | — | block | same as baseline |
| `cifar100_triplet_jepa.json` | 0.5 | class | block | early standalone run |

---

### 2. `latent_triplet` (single encoder + triplet)

```
use_ema_target = False
anchor   = encoder(x_corrupt) OR predictor(encoder(x_corrupt))   # anchor_mode
positive = stopgrad(encoder(x_clean))
negative = scramble / class / instance (if λ > 0)

L = L_jepa + λ · L_triplet     # triplet uses L2-normalized cosine geometry
```

| Axis | Options |
|------|---------|
| `anchor_mode` | `encoder_corrupt`, `predictor_corrupt` |
| `negative_mode` | `instance`, `class`, `scramble`, `scramble_class` |

**Presets / configs:**

| Preset | λ | Anchor | Negatives | Tested? |
|--------|---|--------|-----------|---------|
| `single_encoder_scramble` | 0.05 | encoder | scramble | ✅ k-NN 0.043 @ ep100 |
| `latent_triplet` | 0.05 | encoder | scramble_class | ✅ weak @ ep10 (0.051) |
| `latent_triplet_jepa` | 0 | encoder | scramble | 🔲 control not run |
| `single_encoder_jepa` | 0 | encoder | instance | 🔲 control not run |
| `latent_triplet_heavy` | 25 | encoder | scramble_class | 🔲 ablation not run |
| `latent_triplet_align_heavy` | 25 | predictor | scramble_class | 🔲 ablation not run |
| `cifar100_triplet_scramble.json` | 0.05 | predictor (EMA path) | scramble | ✅ EMA scramble runs |

---

### 3. `latent_vicreg` (single encoder + VICReg)

```
use_ema_target = False
invariance: MSE(predictor(corrupt), encoder(clean))  OR  MSE(encoder(corrupt), encoder(clean))
regularizer: L_var + L_cov on encoder(corrupt) and encoder(clean)
JEPA cosine logged as jepa(log), not optimized when VICReg weights active

L = α·MSE(inv) + μ·L_var + ν·L_cov
```

Default weights in presets: **α=μ=ν=25**.

| Axis | Options |
|------|---------|
| `anchor_mode` | `encoder_corrupt`, `predictor_corrupt` |
| `corrupt_schedule` | `block`, `mask_curriculum` (patch-blur ramp) |

**Presets / configs:**

| Preset | Anchor | Corrupt | VICReg terms | Tested? |
|--------|--------|---------|--------------|---------|
| `latent_vicreg_encoder` | encoder | block | var+cov (inv=25 in preset*) | ✅ 0.129/0.105 @ ep10 |
| `latent_vicreg_align` | predictor | block | inv+var+cov | ✅ 0.130 @ ep10, regressed @ ep20 |
| `latent_vicreg_align_curriculum` | predictor | mask_curriculum | inv+var+cov | ✅ **0.157/0.119** @ ep10 (v2 full VICReg) |
| `latent_vicreg_encoder_curriculum` | encoder | mask_curriculum | inv+var+cov | 🔲 not run |
| v1 curriculum (var+cov only) | predictor | mask_curriculum | var+cov | ⚠️ crashed @ ep21, 0.131 @ ep10 |

\*Early `latent_vicreg.json` run may have used var+cov only before full inv weights were wired.

---

### 4. `latent_sigreg` (single encoder + LeJEPA SIGReg)

```
use_ema_target = False
invariance: MSE(predictor(corrupt), encoder(clean).detach())  OR  encoder path
SIGReg on concat(encoder(corrupt), encoder(clean))

L = (1-λ)·MSE(inv) + λ·SIGReg        # λ = sigreg_weight, default 0.05
jepa(log) = cosine, monitoring only
```

**Presets / configs:**

| Preset | Anchor | Corrupt | Tested? |
|--------|--------|---------|---------|
| `latent_sigreg_align` | predictor | block | ⏳ ep1 only |
| `latent_sigreg_align_curriculum` | predictor | mask_curriculum | ⏳ ep1 only |
| `latent_sigreg_encoder` | encoder | block | 🔲 not run |

---

### 5. `latent_uniformity` (single encoder + hypersphere) — current

```
use_ema_target = False
align: cosine(predictor(corrupt), encoder(clean).detach())
uniformity: Wang-Isola spread on concat(encoder(corrupt), encoder(clean))
optional: SwAV prototype swapping (proto_weight > 0, label-free)

L = cosine(align) + w·Uniformity [+ w_p·SwAV]
```

**Presets / configs:**

| Preset | uniformity_weight | proto_weight | Tested? |
|--------|-------------------|--------------|---------|
| `latent_uniformity_align` (w=1.0) | 1.0 | 0 | ✅ k-NN **0.177** @ ep90, linear 0.025 |
| `latent_uniformity_align` (w=0.5) | 0.5 | 0 | ⏳ 0.125/0.071 @ ep10 |
| `latent_uniformity_swav` | 0.5 | 0.5 | 🔲 config ready, not run |

---

### 6. `latent_triplet_uniformity` (new — cosine triplet + uniformity)

```
use_ema_target = False
anchor   = predictor(encoder(x_corrupt))
positive = stopgrad(encoder(x_clean))
negative = stopgrad(encoder(scramble(x_same)))    # label-free same-image distortion

L = cosine(align) + λ·triplet_cosine(anchor, pos, neg) + w·Uniformity
```

Cosine triplet: `relu(d_pos - d_neg + m)` where `d = 1 - cos` on unit sphere.

| Preset | λ triplet | w uniformity | Negative | Tested? |
|--------|-----------|----------------|----------|---------|
| `latent_triplet_uniformity_align` | 0.5 | 0.5 | scramble (forced) | 🔲 ready to run |

**Command:**
```bash
PYTHONPATH=. python -u train.py --config configs/cifar100_latent_triplet_uniformity_align.json
```

---

### Cross-cutting axes (all modes)

| Parameter | Values | Effect |
|-----------|--------|--------|
| `anchor_mode` | `predictor_corrupt`, `encoder_corrupt` | Where JEPA/triplet alignment is applied |
| `corrupt_schedule` | `block`, `mask_curriculum`, `blur_to_mask` | Fixed 60% mask vs ramping corruption |
| `negative_mode` | `instance`, `class`, `scramble`, `scramble_class` | Triplet negative mining (uses labels only for `class` modes) |
| `triplet_weight` | 0, 0.05, 0.5, 25 | Triplet term strength |
| `use_ema_target` | true (`jepa_ema`), false (all `latent_*`) | Second encoder via EMA |

---

### Tested vs implemented summary

| Category | Count | Notes |
|----------|-------|-------|
| **Modes exercised** | 5/6 | New `latent_triplet_uniformity` ready, not run yet |
| **Fully completed (100 ep)** | ~8 runs | EMA suite, single_encoder_scramble, uniformity w=1.0 |
| **Partial / in progress** | ~6 runs | VICReg variants, SIGReg ep1, uniformity w=0.5 |
| **Implemented, never run** | ~10 presets | heavy triplet, SwAV, blur curriculum, SIGReg encoder, etc. |

**Modes tested with meaningful results:**

1. ✅ `jepa_ema` — **best overall** (0.191 k-NN)
2. ✅ `jepa_ema` + triplet — **negative result**
3. ✅ `latent_triplet` — **failed** (single encoder scramble)
4. ✅ `latent_vicreg` — **promising** (0.157 k-NN, balanced linear)
5. ⏳ `latent_sigreg` — **started only**
6. ✅ `latent_uniformity` — **best no-EMA k-NN**, linear weak

---

## Leaderboard (best known)

| Rank | Run | Best k-NN | Best linear | Best epoch | Notes |
|------|-----|-----------|-------------|------------|-------|
| 1 | EMA JEPA (peak) | **0.191** | **0.155** | 50 | Overall best |
| 2 | uniformity w=1.0 | 0.177 | 0.025 | 90 | k-NN strong, linear collapsed |
| 3 | jepa_baseline (suite) | 0.175 | 0.132 | 100 | Official EMA suite |
| 4 | VICReg curriculum + full | 0.157 | 0.119 | 10 | Best single-encoder w/ linear |
| 5 | uniformity w=0.5 | 0.129 | 0.073 | 10 | In progress |
| — | All triplet variants | ≤0.07 | ≤0.07 | — | Negative results |

---

## Phase 1 — EMA JEPA + triplet (Jul 13, 2026)

**Command:**
```bash
PYTHONPATH=. python -u scripts/run_paper_experiments.py \
  --dataset cifar100 --epochs 100 --output-dir ./results/paper_cifar100
```

| Run | Mode | λ | Negatives | k-NN | Linear | Epoch | Status |
|-----|------|---|-----------|------|--------|-------|--------|
| jepa_baseline | `jepa_ema` | 0 | — | 0.175 | 0.132 | 100 | ✅ Complete |
| triplet_instance | `jepa_ema` | 0.5 | instance | 0.049 | 0.050 | 100 | ❌ Failed |
| triplet_class | `jepa_ema` | 0.5 | class | 0.063 | 0.053 | 100 | ❌ Failed |

**Separate EMA JEPA peak run** (not from suite aggregate):

| Run | k-NN | Linear | Epoch | Notes |
|-----|------|--------|-------|-------|
| JEPA EMA (best checkpoint) | **0.191** | **0.155** | 50 | Pure JEPA, fixed mask; best ≠ final |

**Finding:** Triplet on predictor outputs hurt eval on frozen `encoder(clean)`. JEPA-only with EMA is strongest.

---

## Phase 2 — Scramble / single-encoder triplet (Jul 13–14, 2026)

| Run | Config | k-NN | Linear | Epoch | Status |
|-----|--------|------|--------|-------|--------|
| triplet_scramble | `cifar100_triplet_scramble.json` | ~0.04–0.08 | ~0.02–0.07 | 10 | ❌ Poor |
| single_encoder_scramble | `cifar100_single_encoder_scramble.json` | 0.043 | 0.034 | 100 | ❌ Failed |
| latent_triplet + scramble_class | `cifar100_latent_triplet.json` | 0.051 | 0.024 | 10 | ❌ Weak |

**Finding:** Normalized triplet + encoder anchors stable but still far below EMA. Dual negatives (scramble + class) did not help.

---

## Phase 3 — Single encoder + VICReg (Jul 14, 2026)

| Run | Anchor | Corruption | Loss | k-NN | Linear | Epoch | Status |
|-----|--------|------------|------|------|--------|-------|--------|
| latent_vicreg_encoder | encoder | fixed mask | var+cov | 0.129 | 0.105 | 10 | ✅ Partial |
| latent_vicreg_align | predictor | fixed mask | var+cov | 0.130 | 0.106 | 10 | ✅ Partial |
| latent_vicreg_align | predictor | fixed mask | var+cov | 0.112 | 0.092 | 20 | ⚠️ Regressed after ep10 |
| **latent_vicreg_align_curriculum v2** | predictor | patch-blur + full VICReg | inv+var+cov | **0.157** | **0.119** | 10 | ✅ Best VICReg |

**Commands:**
```bash
PYTHONPATH=. python -u train.py --config configs/cifar100_latent_vicreg.json
PYTHONPATH=. python -u train.py --config configs/cifar100_latent_vicreg_align.json
PYTHONPATH=. python -u train.py --config configs/cifar100_latent_vicreg_align_curriculum.json
```

**Finding:** VICReg enables viable single-encoder training (~88% of EMA at ep10). Heavy var/cov (α=β=25) can trade semantics for spread after ep10. Patch-blur curriculum helped VICReg briefly; crashed at ep21 (fixed with box blur).

---

## Phase 4 — SIGReg (Jul 14–15, 2026)

| Run | Config | k-NN | Linear | Epoch | Status |
|-----|--------|------|--------|-------|--------|
| latent_sigreg_align | `cifar100_latent_sigreg_align.json` | — | — | 1 | ⏳ Started only |
| latent_sigreg_align_curriculum | `cifar100_latent_sigreg_align_curriculum.json` | — | — | 1 | ⏳ Not finished |

**Command:**
```bash
PYTHONPATH=. python -u train.py --config configs/cifar100_latent_sigreg_align.json
```

**Finding:** SIGReg + patch-blur curriculum did not improve over simpler setups. Fixed block mask preferred.

---

## Phase 5 — Cosine + uniformity (Jul 15–16, 2026) — current

**Loss:**
```
loss = cosine(predictor(encoder(corrupt)), encoder(clean))
     + uniformity_weight × Uniformity(encoder embeddings)
```

### Run A — `uniformity_weight=1.0`

| Epoch | k-NN | Linear | align | unif |
|-------|------|--------|-------|------|
| 10 | 0.149 | 0.059 | 0.192 | -3.89 |
| 50 | 0.167 | 0.041 | — | — |
| 90 | **0.177** | 0.025 | — | — |

**Finding:** Best single-encoder k-NN; linear probe collapsed (uniformity over-spreads sphere).

### Run B — `uniformity_weight=0.5` (current config)

| Epoch | k-NN | Linear | align | unif | Notes |
|-------|------|--------|-------|------|-------|
| 10 | 0.129 | 0.073 | 0.159 | -3.83 | — |
| 10 | 0.125 | 0.071 | 0.154 | -3.83 | checkpoint_best saved |

**Command:**
```bash
PYTHONPATH=. python -u train.py --config configs/cifar100_latent_uniformity_align.json
```

**Resume:**
```bash
PYTHONPATH=. python -u train.py \
  --config configs/cifar100_latent_uniformity_align.json \
  --resume results/paper_cifar100/latent_uniformity_align/checkpoint_last.pt
```

**Finding:** Lower weight helps linear slightly but k-NN drops. Still weak linear vs EMA.

### Run C — uniformity + SwAV (not run to completion)

| Run | Config | Status |
|-----|--------|--------|
| latent_uniformity_swav | `cifar100_latent_uniformity_swav.json` | 🔲 Ready, not completed |

**Command:**
```bash
PYTHONPATH=. python -u train.py --config configs/cifar100_latent_uniformity_swav.json
```

---

## Open / next experiments

| ID | Idea | Config / notes | Status |
|----|------|----------------|--------|
| N1 | Finish uniformity w=0.5 to ep100 | `cifar100_latent_uniformity_align.json` | ⏳ In progress |
| N2 | Triplet + cosine on hypersphere (class negatives) | Not implemented | 🔲 Planned |
| N3 | Triplet+cosine+uniformity (scramble neg) | `cifar100_latent_triplet_uniformity_align.json` | 🔲 **new mode** |
| N4 | SimCLR baseline | Discussed, deferred | 🔲 |
| N5 | Update RESULTS_CIFAR100.md | Superseded by this file | — |

---

## Key conclusions (so far)

1. **EMA JEPA (λ=0)** remains the overall best: k-NN 0.191 / linear 0.155 @ ep50.
2. **Triplet margin** consistently hurts at λ=0.5 (and heavy λ=25 unstable).
3. **VICReg** is the best single-encoder path for **balanced** k-NN + linear (0.157 / 0.119 @ ep10).
4. **Uniformity** maximizes k-NN without EMA (0.177) but **kills linear probe**.
5. **Predictor alignment head** is kept; EMA dropped from main architecture.
6. **Patch-blur curriculum** did not help SIGReg; marginal for VICReg.
7. **λ / weight sensitivity** is a real weakness — performance depends heavily on regularizer weight.

---

## How to log a new run

Copy this block under the relevant phase (or add a new phase):

```markdown
### YYYY-MM-DD — short_name

**Config:** `configs/....json`
**Command:** `PYTHONPATH=. python -u train.py --config ...`

| Epoch | k-NN | Linear | loss terms... | Notes |
|-------|------|--------|---------------|-------|
| 10 | — | — | — | |
| 100 | — | — | — | final |

**Finding:** one sentence.
```

Also update the **Leaderboard** table if the run beats a prior best.

---

*Last updated: 2026-07-16*
