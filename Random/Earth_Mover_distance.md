# Earth Mover’s Distance (EMD)

## Overview

Earth Mover’s Distance (EMD), also known as the 1-Wasserstein distance, measures the minimal “work” needed to transform one probability distribution into another by transporting probability mass across an underlying metric space. Intuitively, imagine two piles of dirt (histograms) and the cost to move dirt between bins—EMD finds the cheapest way to match them.

### Why EMD Matters in AI

- **Order-sensitive errors**: Unlike cross-entropy, EMD respects the numeric or geometric ordering of labels (e.g., predicting 6 instead of 5 is less costly than predicting 9 instead of 5).
- **Robust generative training**: WGANs and related models use EMD to stabilize adversarial training by measuring generator-discriminator mismatch in a geometry-aware way.
- **Number Token Loss (NTL)**: Recent work adds an EMD-based regression‐style token loss to LLM pretraining, boosting math performance by penalizing numeric predictions proportional to their distance from the true values.

![image](https://github.com/user-attachments/assets/63fb0229-b935-41d0-8a72-45715a289206)
---

## 3. Quick Start: Implementing EMD in Python

```python
import numpy as np
import ot  # POT: Python Optimal Transport

# Define two discrete distributions
P = np.array([0.4, 0.6])
Q = np.array([0.5, 0.3, 0.2])

# Cost matrix based on bin positions
bins_P = np.arange(len(P))[:, None]
bins_Q = np.arange(len(Q))[None, :]
M = np.abs(bins_P - bins_Q)

# Compute squared 1-Wasserstein distance
emd2 = ot.emd2(P, Q, M)
emd = np.sqrt(emd2)
print(f"EMD^2 = {emd2:.4f}, EMD = {emd:.4f}")
```

### Integrating into a Training Loop

```python
# Inside your PyTorch training step:
loss_ce = F.cross_entropy(logits, labels)
loss_em = sinkhorn_distance(pred_softmax, target_softmax, M, reg=0.05)
loss = loss_ce + λ * loss_em
loss.backward()
```

---

## 4. Code Example: Wasserstein Number Token Loss (NTL)

This snippet uses the `WassersteinNumberTokenLoss` class from the [number-token-loss](https://github.com/tum-ai/number-token-loss) repo:

```python
from ntl.loss import WassersteinNumberTokenLoss

# Build and sort number tokens (0–9)
ntl = WassersteinNumberTokenLoss(tokenizer, vocab_size, device, order_numbers=True)

# Compute NTL on model outputs
loss_ntl = ntl.forward(logits, labels)
```

```bash
python run_language_modeling.py \
  model_args.number_token_loss_with_wasserstein=true \
  model_args.number_token_loss_weight=0.3 \
  dataset_args=mathematics_dataset
```

---

## 5. Advanced Applications in GANs and Other Models

### 5.1 Wasserstein GANs (WGAN)

- **Kantorovich–Rubinstein Duality**: WGAN minimizes the 1-Wasserstein distance between real and generated data distributions via a critic network with Lipschitz constraint.
- **Loss formulation**:
  $$
    \min_G \max_{D\in\mathcal{D}} \;\mathbb{E}_{x\sim P_r}[D(x)] - \mathbb{E}_{z\sim P_z}[D(G(z))]
  $$
  where \(\mathcal{D}\) enforces 1-Lipschitz continuity (originally via weight clipping, later via gradient penalty).
- **Improvements**:
  - WGAN-GP (2017): replaces weight clipping with an explicit gradient penalty for stable training (Gulrajani et al.).
  - Spectral Normalization (Miyato et al.): enforces Lipschitz condition by controlling singular values of weight matrices.

### 5.2 EMD in Attention and Ranking

- **Differentiable sorting**: Recent methods use EMD to define losses over permutations or rank lists, enabling end-to-end training for tasks like summarization ranking and differentiable top-k selection.
- **Document retrieval**: EMD between query and document term distributions captures semantic shifts better than simple cosine or KL measures.

### 5.3 Recent Research Highlights

| Application                | Paper / Year                 | Usage of EMD                                                                              |
| -------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------- |
| Point Cloud Registration   | Fu et al., CVPR 2021         | EMD between 3D point sets for alignment loss                                              |
| Domain Adaptation          | Courty et al., NeurIPS 2017  | EMD to measure distribution shift between source and target features                      |
| Medical Image Segmentation | Kervadec et al., MICCAI 2020 | Sinkhorn-based EMD loss for aligning predicted soft masks with ground truth segmentations |
| Scene Graph Modeling       | Zhao et al., ICCV 2021       | EMD to compare object attribute distributions across scene graph nodes                    |

---

## 6. Appendix

### A. Mathematical Derivation of 1D EMD

Given discrete distributions \(P=(p_1,\dots,p_n)\) and \(Q=(q_1,\dots,q_n)\) over ordered values \(v_1<\cdots<v_n\):

1. **Cumulative distributions**: \(F_P(k)=\sum_{i=1}^k p_i, \quad F_Q(k)=\sum_{i=1}^k q_i.\)
2. **1-Wasserstein distance**: \(W_1(P,Q)=\sum_{k=1}^n |F_P(k)-F_Q(k)|.\)

This formula equals the solution of the optimal transport linear program \(
\min_{T_{ij}\ge0}\sum_{i,j}T_{ij}d(v_i,v_j)\) subject to \(\sum_jT_{ij}=p_i, \sum_iT_{ij}=q_j.\)

### B. Worked Example

Consider \(P=(0.2,0.5,0.3)\) at bins \(\{1,2,3\}\) and \(Q=(0.4,0.4,0.2)\):

1. CDFs: \(F_P=(0.2,0.7,1.0), F_Q=(0.4,0.8,1.0)\).
2. Differences: \(|0.2-0.4|+|0.7-0.8|+|1.0-1.0|=0.2+0.1+0=0.3\).

Thus, \(EMD(P,Q)=0.3\).

### C. AI Integration Details

- **Softmax extraction**: select only the number-token logits and compute probabilities.
- **Cost matrix**: build using absolute differences of the decoded token values.
- **Differentiable OT**: use Sinkhorn for backprop-friendly training.

---

---

## 7. EMD as a General ML Training Objective

Beyond specific applications like GANs and number-token losses, EMD can serve as a versatile loss in various ML tasks where outputs or labels lie on a metric or ordered domain:

### 7.1 Regression and Distribution Matching

- **Histogram regression**: When predicting binned quantities (e.g., age ranges, risk scores), replace MSE on bin indices with EMD to penalize predictions proportionally to bin distance.
- **Calibration of probabilistic forecasts**: Match predicted probability histograms (e.g., ensemble forecasts, uncertainty estimates) to empirical distributions using Sinkhorn-based OT, yielding better-calibrated models.

```python
# Example: age-group prediction
pred_hist = model_output  # shape [batch, num_bins]
true_hist = one_hot_age_bins  # same shape
loss_emd = torch.sum(torch.abs(pred_hist.cumsum(-1) - true_hist.cumsum(-1)), dim=-1).mean()
```

### 7.2 Ordinal and Cost-Sensitive Classification

- **Ordinal labels** (e.g., star ratings, medical stages): Standard cross-entropy treats all misclassifications equally; EMD penalizes by label distance, improving performance on adjacent-class errors.
- **Cost-sensitive learning**: Define a ground metric reflecting application costs (e.g., misdiagnosis severity), then use EMD to minimize expected cost directly.

### 7.3 Structured Prediction & Sequence Tasks

- **Sequence alignment**: Compare length-normalized histograms of n‑gram features between generated and reference sequences, using EMD to guide language models or summarization systems.
- **Graph-structured outputs**: For tasks predicting distributions over nodes (e.g., scene graphs, molecular graphs), use EMD over node or feature embeddings to align predicted graphs to ground truth.

### 7.4 Integration Tips

1. **Choose the right granularity**: Binning and tokenization should correspond to problem semantics (e.g., quantile bins for continuous targets).
2. **Compute cost matrix**: Use meaningful ground distances (Euclidean, domain-specific distances). Precompute once if static.
3. **Regularize for speed**: Use Sinkhorn with a small entropy term (0.01–0.1) for large-scale tasks to balance bias and efficiency.
4. **Combine with auxiliary losses**: Often best paired with standard objectives (e.g., MSE, CE) via weighted sum.

```python
# Combined loss example
loss = α * ce_loss + β * emd_loss + γ * mse_loss
```

---

---

## 8. Example: Incorporating EMD in XGBoost

EMD can be integrated into XGBoost workflows either as a custom evaluation metric or, more advanced, as a custom objective for ordinal or histogram regression tasks.

### 8.1 Using EMD as a Custom Evaluation Metric

Below is a simple example where we train an XGBoost model for predicting a histogram over 5 ordered bins (e.g., risk scores) and evaluate with the 1-Wasserstein distance using SciPy's implementation:

```python
import xgboost as xgb
import numpy as np
from scipy.stats import wasserstein_distance

# Suppose y_train are true bin indices (0–4)
bins = 5
y_one_hot = np.eye(bins)[y_train]

# Prepare DMatrix with multi-class softprob
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'objective': 'multi:softprob',
    'num_class': bins,
    'eval_metric': 'mlogloss'
}

# Custom EMD evaluation function
def emd_eval(preds, dmatrix):
    # preds: flat array, reshape to [n_samples, bins]
    preds = preds.reshape(-1, bins)
    labels = dmatrix.get_label().astype(int)
    true = np.eye(bins)[labels]
    emds = [wasserstein_distance(preds[i], true[i]) for i in range(len(labels))]
    return 'emd', float(np.mean(emds))

# Train with custom metric
evals = [(dtrain, 'train')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evals,
    feval=emd_eval,
    maximize=False
)
```

This reports the average EMD on the training set at each boosting round. You can also pass `feval` under `xgb.cv` or for early stopping.

### 8.2 As a Custom Objective for Ordinal Regression

For more fine-grained control, you can implement a custom objective that approximates gradients of the EMD loss. For instance, for a 1D CDF-based loss:

```python
import xgboost as xgb
def emd_obj(preds, dmatrix):
    bins = preds.shape[1]
    # Apply softmax to get probabilities
    preds = preds.reshape(-1, bins)
    exp_preds = np.exp(preds)
    prob_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    labels = dmatrix.get_label().astype(int)
    true = np.eye(bins)[labels]

    # Compute CDFs
    cdf_p = np.cumsum(prob_preds, axis=1)
    cdf_t = np.cumsum(true, axis=1)
    diff = cdf_p - cdf_t

    # Approximate gradient: difference of PDFs derived from CDF difference
    grad = np.sign(diff)
    # Approximate Hessian as ones
    hess = np.ones_like(grad)

    # Flatten back
    return grad.flatten(), hess.flatten()

# Then train with:
bst = xgb.train(
    {'num_class': bins},
    dtrain,
    num_boost_round=200,
    obj=emd_obj
)
```

This custom objective is a heuristic: it uses CDF differences to shape the gradient toward minimizing EMD. In practice, you may refine the gradient/hessian calculation for stability.


