# Initial experiments (archive)

Moved here from `configs/` — first exploration phase (SmallResNet / hybrids / ablations).

**Paper line:** `../cifar100_jepa_vit_tiny.json` (ViT-Tiny + JEPA).

## Notable runs

| Config | Role |
|--------|------|
| `cifar100_latent_infonce_jepa_sigreg.json` | InfoNCE + JEPA mask + SIGReg (SmallResNet) |
| `cifar100_latent_infonce_jepa_mse_var_cov.json` | InfoNCE + MVC hybrid |
| `cifar100_latent_infonce_jepa_sigreg_resnet50.json` | Same hybrid, ResNet-50 |
| `cifar100_single_encoder_jepa.json` | Pure cosine JEPA (no EMA) |
| `cifar100_jepa.json` | EMA JEPA baseline |
| `cifar100_latent_infonce_jepa_augment.json` | Dual InfoNCE |

Usage:

```bash
PYTHONPATH=. python train.py --config configs/initial_exps/<name>.json --device cuda:0
```
