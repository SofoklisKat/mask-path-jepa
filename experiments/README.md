# Experiment tracking

| Path | Purpose |
|------|---------|
| `experiments/logs/` | Timestamped stdout (gitignored) |
| `results/<run_name>/` | Checkpoints and `results.json` (gitignored) |

```bash
bash scripts/run_experiment.sh <experiment_id> <config.json> <gpu> [extra args]
```

Paper method:

```bash
bash scripts/run_experiment.sh mask_path configs/cifar10_mask_path_infonce.json 0
```
