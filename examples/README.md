# Examples

## Mask-path InfoNCE on MNIST

Single encoder, no EMA: InfoNCE on `predictor(masked)` vs stop-grad `encoder(clean)`.

```bash
pip install -r ../requirements.txt
python mask_path_infonce_mnist.py --epochs 5
```

The predictor is training-only. Evaluation uses the frozen encoder (5-NN).
