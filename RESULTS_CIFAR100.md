# CIFAR-100 Results — Triplet-JEPA Study

Final numbers from the main experiment suite (`batch_size=256`, `epochs=100`, `seed=42`).

## EMA + predictor (default protocol)

| Method | λ | Negatives | k-NN@20 | Linear probe | Notes |
|--------|---|-----------|---------|--------------|-------|
| **jepa_baseline** | 0.0 | — | **0.175** | **0.132** | Best result |
| triplet_instance | 0.5 | instance | 0.049 | 0.050 | Hurts vs baseline |
| triplet_class | 0.5 | class | 0.063 | 0.053 | Hurts vs baseline |

Command:

```bash
PYTHONPATH=. python -u scripts/run_paper_experiments.py \
  --dataset cifar100 \
  --epochs 100 \
  --output-dir ./results/paper_cifar100
```

## Additional ablations

| Method | λ | Anchor | EMA | Negatives | k-NN@20 | Linear probe |
|--------|---|--------|-----|-----------|---------|--------------|
| triplet_scramble | 0.05 | predictor_corrupt | yes | scramble | ~0.04–0.08 @ ep10 | ~0.02–0.07 @ ep10 |
| single_encoder_scramble | 0.05 | encoder_corrupt | no | scramble | 0.043 | 0.034 |

Configs:

```bash
# EMA + scramble (lower λ)
PYTHONPATH=. python -u train.py --config configs/cifar100_triplet_scramble.json

# Single encoder, no EMA
PYTHONPATH=. python -u train.py --config configs/cifar100_single_encoder_scramble.json

# Single encoder JEPA-only control (recommended if not run yet)
PYTHONPATH=. python -u train.py --config configs/cifar100_single_encoder_jepa.json
```

## Conclusions (study complete)

1. **JEPA-only (λ=0) with EMA target is the strongest method** on CIFAR-100 under this budget.
2. **Triplet margin (λ=0.5) consistently degrades** k-NN and linear probe, regardless of negative mining (`instance`, `class`).
3. **Scramble negatives** (same-image patch shuffle) did not rescue performance; single-encoder latent loss performed worse.
4. **Triplet on predictor outputs** optimizes a different space than frozen `encoder(clean)` used at eval — likely explanation for the gap.
5. **Normalized triplet loss** (unit-sphere distances) is required for stable training when using `encoder_corrupt` anchors.

## Paper takeaway

> Adding a triplet margin term to JEPA latent prediction does **not** improve CIFAR-100 representation quality at equal compute in our low-resource setting. JEPA regression alone is the viable baseline; triplet variants are reported as negative results.

## Next study (deferred)

SimCLR-style instance contrastive baseline for a more explicit latent-space contract.
