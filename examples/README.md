# Triplet-JEPA examples

## Minimal MNIST demo

```bash
pip install -r ../requirements.txt
python triplet_jepa_minimal.py --epochs 5
```

### What it demonstrates

| Role | Tensor | Source |
|------|--------|--------|
| **Anchor** | `z_anchor` | `predictor(encoder(corrupt))` |
| **Positive** | `z_positive` | `target_encoder(clean)` (stop-grad, EMA) |
| **Negative** | `z_negative` | shuffled batch positives (other images) |

Combined loss:

```
L = L_jepa + λ * L_triplet
L_jepa     = mean(1 - cos(z_anchor, z_positive))
L_triplet  = relu(||a-p||² - ||a-n||² + margin)
```

The predictor is training-only; at eval time only `encoder` is used (5-NN probe).

See `ssl_jepa_triplet/NOTES.md` for paper context and extension ideas (patch triplets, class labels, Proxy-NCA).
