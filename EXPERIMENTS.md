# Experimental protocol

Matched recipe used in the technical report. Full write-up: [`paper/main.tex`](paper/main.tex).

## Shared settings

| Setting | Value |
|---------|-------|
| Datasets | CIFAR-10 (primary, 3 seeds), CIFAR-100 (exploratory, seed 42) |
| Backbone | ResNet-18, CIFAR stem, embedding dim 512 |
| Predictor | MLP on the mask path (discarded at eval) |
| Masking | Multi-block, ratio 0.6 |
| Optimizer | Adam, lr `3e-4`, batch 256, 100 epochs |
| InfoNCE | τ = 0.1, stop-gradient on keys |
| Metrics | Frozen k-NN@20 (cosine) and linear probe |

Except for the EMA reference, all SSL runs use **one online encoder**.

## Paper runs

```bash
# Proposed method
PYTHONPATH=. python train.py --config configs/cifar10_mask_path_infonce.json --download

# EMA cosine JEPA
PYTHONPATH=. python train.py --config configs/cifar10_jepa_resnet18.json

# Augmentation-path InfoNCE
PYTHONPATH=. python train.py --config configs/cifar10_latent_infonce_jepa.json

# Regularizer hybrids
PYTHONPATH=. python train.py --config configs/cifar10_jepa_vicreg.json
PYTHONPATH=. python train.py --config configs/cifar10_jepa_sigreg.json
PYTHONPATH=. python train.py --config configs/cifar10_jepa_mse_var_cov.json

# Supervised upper bound
PYTHONPATH=. python train_supervised.py --config configs/cifar10_supervised_resnet18.json
```

Report the **fixed epoch-100 endpoint**. Seeds for the CIFAR-10 table: 42, 7, 123
(`--` override via JSON `seed`, or edit the config).

Outputs go to `results/<run_name>/` (`results.json`, checkpoints). That directory
is gitignored.
