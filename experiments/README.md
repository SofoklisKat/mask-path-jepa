# Experiment tracking

## Local tracking (default, no account)

This repo keeps a lightweight registry:

| Path | Purpose |
|------|---------|
| `experiments/runs.jsonl` | One JSON line per run (id, config, GPU, log path, status) |
| `experiments/logs/` | Archived stdout logs (timestamped, kept when runs stop) |
| `results/<run_name>/` | Checkpoints, `config.json`, `results.json` with full metric history |

Launch with:

```bash
bash scripts/run_experiment.sh <experiment_id> <config.json> <gpu> [extra args]
```

Example:

```bash
bash scripts/run_experiment.sh jepa_only configs/cifar100_single_encoder_jepa.json 1
```

View runs:

```bash
cat experiments/runs.jsonl | python3 -m json.tool
tail -f experiments/logs/<latest>.log
```

## Free external tools (optional)

| Tool | Cost | Best for |
|------|------|----------|
| **[MLflow](https://mlflow.org/)** | Free, local or self-hosted | Metric curves, compare runs, artifact store |
| **[Weights & Biases](https://wandb.ai/)** | Free personal tier | Live dashboards, shareable links |
| **[TensorBoard](https://www.tensorflow.org/tensorboard)** | Free | Scalar plots if training logs TB events |
| **[Neptune](https://neptune.ai/)** | Free tier (limited) | Team experiment tables |

**Recommendation for this project:** keep `experiments/runs.jsonl` + `results.json` history (already written each run). Add **MLflow** later if you want a UI without cloud lock-in:

```bash
pip install mlflow
mlflow ui --backend-store-uri ./mlflow
```

W&B is the fastest upgrade if you want live plots with ~5 lines of code in `train.py` (`wandb.init`, `wandb.log` each epoch).
