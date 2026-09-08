# Mask-Path InfoNCE without EMA in a Global Image-Level JEPA

**A CIFAR study** — technical report (2026).

Joint-Embedding Predictive Architectures (JEPAs) usually stabilize latent
prediction with an EMA teacher. This repository studies a simpler alternative:
apply **InfoNCE on the masked predictive pair** of a **single** ResNet-18, with
stop-gradient targets and no EMA.

Paper: [`paper/main.pdf`](paper/main.pdf) · source: [`paper/main.tex`](paper/main.tex)

## Results

CIFAR-10, ResNet-18, 100 epochs, frozen encoder. Mean ± sample std over seeds 42, 7, and 123.

| Method | k-NN@20 | Linear probe |
|--------|---------|--------------|
| **Mask-path InfoNCE (no EMA)** | **42.6 ± 3.6%** | **43.0 ± 3.4%** |
| EMA cosine JEPA | 41.6 ± 2.9% | 41.3 ± 2.7% |
| Cosine JEPA + aug-path InfoNCE (no EMA) | 40.2 ± 1.0% | 39.9 ± 1.9% |
| Cosine JEPA + VICReg (no EMA) | 37.4 ± 0.4% | 39.3 ± 0.7% |
| Supervised CE (upper bound) | 94.9 ± 0.1% | 94.8 ± 0.1% |

The mask-path means are slightly higher than EMA, but the seed ranges overlap.
CIFAR-100 (seed 42 only) shows the same ranking: 15.5% k-NN vs 12.5% for EMA.

## Method

```
z_c     = encoder(clean)
ẑ_m    = predictor(encoder(masked))     # mask ratio 0.6
L_NCE   = InfoNCE(ẑ_m, sg(z_c); τ = 0.1)
```

One encoder, no EMA. The predictor is discarded at evaluation. Contrast is
applied to the **mask path**, not to a separate augmentation view.

## Quick start

```bash
pip install -r requirements.txt

# Tiny MNIST demo (mask-path InfoNCE, no EMA)
python examples/mask_path_infonce_mnist.py --epochs 5

# Paper method on CIFAR-10
PYTHONPATH=. python train.py --config configs/cifar10_mask_path_infonce.json --download
```

Frozen-encoder evaluation is built into training: **k-NN@20** (cosine) and a
linear probe on the standard CIFAR split.

## Paper configs

| Config | Role in the report |
|--------|--------------------|
| `configs/cifar10_mask_path_infonce.json` | Proposed: mask-path InfoNCE, no EMA |
| `configs/cifar10_jepa_resnet18.json` | EMA cosine JEPA reference |
| `configs/cifar10_latent_infonce_jepa.json` | Cosine JEPA + augmentation-path InfoNCE |
| `configs/cifar10_jepa_vicreg.json` | Cosine JEPA + VICReg |
| `configs/cifar10_jepa_sigreg.json` | Cosine JEPA + SIGReg |
| `configs/cifar10_jepa_mse_var_cov.json` | Cosine JEPA + MSE/var/cov |
| `configs/cifar10_supervised_resnet18.json` | Supervised CE upper bound |
| `configs/cifar100_mask_path_infonce.json` | CIFAR-100 mask-path run (single seed) |

Earlier exploration configs live under `configs/initial_exps/`.

## Citation

GitHub reads [`CITATION.cff`](CITATION.cff) and offers a **Cite this repository**
button (APA and BibTeX). For papers, use:

```bibtex
@misc{katakis2026maskpath,
  title        = {Mask-Path InfoNCE without EMA in a Global Image-Level JEPA: A CIFAR Study},
  author       = {Katakis, Sofoklis},
  year         = {2026},
  howpublished = {Technical report},
  url          = {https://github.com/SofoklisKat/mask-path-jepa/blob/main/paper/main.pdf}
}
```

## Layout

```
mask_path_jepa/   # encoder, losses, training, evaluation
configs/          # paper configs + archived explorations
paper/            # report PDF (main.pdf) and LaTeX source
examples/         # MNIST mask-path demo
scripts/          # download, plots, experiment launchers
```

## License

MIT. See [LICENSE](LICENSE).
