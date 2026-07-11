# TripletJEPA

**Margin-regularized Joint-Embedding Predictive Architectures** for self-supervised image representations.

Combines JEPA latent prediction (corrupt → clean embedding) with triplet margin loss for better embedding geometry — designed for **low-resource, paper-grade** experiments on CIFAR-10.

## Datasets

| Dataset | Classes | Size | Download | Config |
|---------|---------|------|----------|--------|
| `cifar10` | 10 | 32×32 | `--download` | `configs/cifar10_triplet_jepa.json` |
| `cifar100` | 100 | 32×32 | `--download` (~169 MB) | `configs/cifar100_triplet_jepa.json` |
| `tiny_imagenet` | 200 | 64×64 | `--download` (~237 MB) | `configs/tiny_imagenet_triplet_jepa.json` |
| `imagenet` | 1000 | resized 64×64 | manual copy | `configs/imagenet_triplet_jepa.json` |
| `mnist` | 10 | 28×28 | `--download` | — |

Downloads are **off by default**. Use `--download` when training, or download first:

```bash
# Download only (recommended first step)
PYTHONPATH=. python scripts/download_dataset.py --dataset cifar100

# Train with download if missing
PYTHONPATH=. python train.py --config configs/cifar100_triplet_jepa.json --download
```

```bash
# CIFAR-100
PYTHONPATH=. python scripts/download_dataset.py --dataset cifar100
PYTHONPATH=. python train.py --config configs/cifar100_triplet_jepa.json

# Tiny ImageNet
PYTHONPATH=. python scripts/download_dataset.py --dataset tiny_imagenet
PYTHONPATH=. python train.py --config configs/tiny_imagenet_triplet_jepa.json

# ImageNet — place data at data/imagenet/train/<class>/ and data/imagenet/val/<class>/
PYTHONPATH=. python train.py --config configs/imagenet_triplet_jepa.json
```

Full ImageNet is ~150 GB; the config uses 64×64 resize + optional `train_subset` for low-resource runs.

## Quick start

```bash
pip install -r requirements.txt

# Validate pipeline (~5 min)
python train.py --config configs/quick_smoke.json

# Train Triplet-JEPA on CIFAR-10
python train.py --config configs/cifar10_triplet_jepa.json

# Run all baselines + generate results table
python scripts/run_paper_experiments.py --quick   # fast
python scripts/run_paper_experiments.py         # full (100 epochs)
```

## Method

```
L = L_JEPA + λ · L_triplet
L_JEPA     = 1 - cos(predictor(encoder(corrupt)), target_encoder(clean))
L_triplet  = relu(||a-p||² - ||a-n||² + margin)
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full protocol and [ssl_jepa_triplet/NOTES.md](ssl_jepa_triplet/NOTES.md) for literature notes.

## Project layout

```
tripletjepa/          # library (models, losses, train, eval)
configs/              # JSON experiment configs
scripts/              # paper experiment suite
examples/             # minimal MNIST tutorial
EXPERIMENTS.md        # reproducible paper protocol
```

## Evaluation

Frozen encoder only:
- **k-NN@20** (cosine, DINO-style)
- **Linear probe** (SGD classifier on train features)

## Citation

```bibtex
@misc{tripletjepa2026,
  title={Margin-Regularized Joint-Embedding Predictive Architectures},
  author={...},
  year={2026},
  note={Challenges Paper 2026}
}
```
