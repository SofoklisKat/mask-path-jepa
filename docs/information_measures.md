# Measurements of Information

A short reference for quantities used in SSL, embeddings, and “how much does this representation tell us?”  
Notation uses plain text so it renders everywhere.

---

## 1. Entropy — H(X)

**Meaning:** Uncertainty / average surprise of a random variable X.  
**Units:** bits (log base 2) or nats (natural log).

For a discrete variable with probabilities p(x):

```
H(X) = - sum_x  p(x) * log p(x)
```

- H = 0 → deterministic (no uncertainty)
- H large → many outcomes, roughly equal probability

### Example

Fair coin: p(H)=p(T)=1/2

```
H = - (0.5 log2 0.5 + 0.5 log2 0.5) = 1 bit
```

Biased coin p(H)=0.9, p(T)=0.1:

```
H ≈ 0.47 bits   (less uncertainty)
```

**CIFAR-100 class label Y (100 classes, uniform):**

```
H(Y) = log2(100) ≈ 6.64 bits
```

---

## 2. Joint entropy — H(X, Y)

Uncertainty of the pair (X, Y):

```
H(X, Y) = - sum_{x,y}  p(x,y) * log p(x,y)
```

Always:

```
H(X, Y) ≤ H(X) + H(Y)
```

Equality iff X and Y are independent.

---

## 3. Conditional entropy — H(Y | X)

Uncertainty left in Y after observing X:

```
H(Y | X) = H(X, Y) - H(X)
```

- H(Y|X) = 0 → X determines Y perfectly  
- H(Y|X) = H(Y) → X tells nothing about Y  

### Example (toy)

Y = image class (2 classes, fair).  
X = a feature that is correct 80% of the time.

Roughly H(Y) = 1 bit, H(Y|X) ≈ 0.72 bits  
→ feature removes about 0.28 bits of class uncertainty.

---

## 4. Mutual information — I(X; Y)

**Meaning:** How much knowing X reduces uncertainty about Y (shared information).

```
I(X; Y) = H(Y) - H(Y | X)
        = H(X) - H(X | Y)
        = H(X) + H(Y) - H(X, Y)
```

Properties:

- I(X; Y) ≥ 0  
- I(X; Y) = 0 iff independent  
- Symmetric: I(X; Y) = I(Y; X)

### Example (SSL intuition)

- X = image embedding z  
- Y = class label  

High I(z; Y) → embedding is **semantically informative** (good linear probe / same-class neighbors).  
Your neighbor grids suggest I(z; Y) is still **small** on CIFAR-100 for current checkpoints.

### Example (numbers)

Suppose H(Y) = 6.64 bits (CIFAR-100).  
If a perfect classifier exists from z, then H(Y|z) = 0 and I(z; Y) = 6.64.  
If linear probe accuracy is weak, I(z; Y) is much smaller (exact value needs density estimation).

---

## 5. Pointwise mutual information — PMI

For one pair (x, y):

```
PMI(x; y) = log( p(x,y) / (p(x) p(y)) )
```

- PMI > 0 → x and y co-occur more than chance  
- Mutual information is the expectation of PMI:

```
I(X; Y) = E[ PMI(X; Y) ]
```

---

## 6. Kullback–Leibler divergence — KL(P || Q)

**Meaning:** Extra bits to code samples from P using a code built for Q.  
Not symmetric.

```
KL(P || Q) = sum_x  p(x) * log( p(x) / q(x) )
```

### Example

P = true label distribution, Q = model softmax.  
Minimizing KL(P || Q) is equivalent (up to constant) to minimizing cross-entropy.

**Link to SIGReg / Gaussian priors:**  
SIGReg pushes embedding statistics toward a target (e.g. isotropic Gaussian) — related in spirit to reducing divergence from that prior.

---

## 7. Cross-entropy — H(P, Q)

```
H(P, Q) = - sum_x  p(x) * log q(x)
         = H(P) + KL(P || Q)
```

Classification training loss ≈ cross-entropy between one-hot labels and predictions.

---

## 8. Conditional mutual information — I(X; Y | Z)

Information shared by X and Y **beyond** what Z already explains:

```
I(X; Y | Z) = H(Y | Z) - H(Y | X, Z)
```

### Example (your physical / geometric prior idea)

- X = color statistics of top of image  
- Y = “sky-like” region label  
- Z = spatial layout (top vs bottom)  

I(X; Y | Z) asks: after knowing **where**, how much does **appearance** still tell?

---

## 9. Information bottleneck (IB)

Learn representation Z of input X that keeps info about task Y but compresses X:

```
minimize   I(X; Z) - β * I(Z; Y)
```

- Small I(X; Z) → discard nuisance detail  
- Large I(Z; Y) → keep semantics  

SSL often has **no labels Y** at train time, so people use **proxy tasks** (augmented views, masks) instead of I(Z; Y).

---

## 10. InfoNCE as a mutual-information estimator

InfoNCE (SimCLR-style) is a **lower bound** on mutual information between two views:

```
I(view_a; view_b)  ≥  log(B) - L_InfoNCE
```

where B = batch size (number of negatives + 1).

### Example

Batch size 256:

```
log2(256) = 8 bits
```

If InfoNCE loss ≈ 1.0 nat, convert carefully; the bound says MI is at least roughly on the order of several bits between views — **not** the same as MI with the class label.

So: low InfoNCE ≠ “semantic neighborhoods are correct.” Your NN grids show that gap.

---

## 11. Redundancy measures (embeddings)

For a batch of embeddings Z (N × D), e.g. 100 × 256:

| Measure | Idea |
|---------|------|
| **Covariance off-diagonals** | correlated dims = redundant channels (VICReg cov term) |
| **Effective rank / PCA spectrum** | how many axes actually carry variance |
| **Total correlation** | sum_i H(Z_i) - H(Z) — shared info among dimensions |

### Example (PCA effective dimension)

If 90% of variance lives in the first 40 principal components of a 256-D embedding, the code is heavily **redundant / compressible** to ~40 dims for that criterion.

---

## 12. Distortion / rate (geometry + information)

Rate–distortion: encode X into Z with rate R = I(X; Z), minimize expected distortion D(X, decode(Z)).

JEPA-style losses are closer to **predictive distortion in latent space** than to pixel MSE:

```
distortion ≈ distance( predictor(mask), stopgrad(clean) )
```

---

## Cheat sheet

| Symbol | Name | One-line meaning |
|--------|------|------------------|
| H(X) | Entropy | Uncertainty of X |
| H(Y\|X) | Conditional entropy | Uncertainty left in Y given X |
| I(X; Y) | Mutual information | Shared information / dependence |
| KL(P\|\|Q) | KL divergence | How different P is from Q |
| H(P,Q) | Cross-entropy | Coding cost of P under Q |
| InfoNCE | Contrastive bound | Trainable lower bound on I(views) |

---

## Tiny worked example (2 bits)

Two binary variables A, B:

```
P(A=0,B=0)=0.4
P(A=0,B=1)=0.1
P(A=1,B=0)=0.1
P(A=1,B=1)=0.4
```

Then:

```
P(A=0)=0.5, P(A=1)=0.5  →  H(A)=1 bit
P(B=0)=0.5, P(B=1)=0.5  →  H(B)=1 bit

H(A,B) = - (0.4 log2 0.4 + 0.1 log2 0.1 + 0.1 log2 0.1 + 0.4 log2 0.4)
       ≈ 1.72 bits

I(A;B) = H(A)+H(B)-H(A,B) ≈ 2 - 1.72 = 0.28 bits
```

A and B are **weakly dependent** (same value more often than chance).

---

## Link to this project

| Goal | Information view |
|------|------------------|
| Same-class neighbors | Maximize I(z; class) without using labels at train time |
| Diagonal JEPA (clean ↔ mask) | Maximize I(clean_view; mask_view) / predictive agreement |
| Avoid collapse | Keep H(z) / variance high; reduce dimension-wise redundancy |
| Geometric priors | Soft constraints so z retains layout/structure info, not only instance ID |

---

## References (classic)

- Cover & Thomas, *Elements of Information Theory*  
- InfoNCE: Oord et al., “Representation Learning with Contrastive Predictive Coding”  
- VICReg: Bardes et al. (variance–invariance–covariance ≈ anti-redundancy)  
- Information bottleneck: Tishby et al.
