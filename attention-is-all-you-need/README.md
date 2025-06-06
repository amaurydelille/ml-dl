# Mathematics Behind the Attention Mechanism

## Goal of Attention

The attention mechanism allows a model to **dynamically focus** on relevant parts of the input sequence when making predictions. It computes a **weighted sum of values**, where the weights are determined by the similarity between queries and keys.

---

## Notation and Dimensions

Let:
- $Q \in \mathbb{R}^{n_q \times d_k}$: Query matrix (e.g., from decoder or self-attention)
- $K \in \mathbb{R}^{n_k \times d_k}$: Key matrix (e.g., from encoder or same input)
- $V \in \mathbb{R}^{n_k \times d_v}$: Value matrix (contextual representations)
- $d_k$: Dimensionality of keys/queries
- $d_v$: Dimensionality of values
- $n_q$: Number of query vectors
- $n_k$: Number of key/value vectors

---

## 1. Scaled Dot-Product Attention

Given query, key, and value matrices, attention is computed in the following steps:

### Step 1: Compute Compatibility (Dot Product)

$$
\text{Scores} = QK^T \in \mathbb{R}^{n_q \times n_k}
$$

Each element $\text{Scores}_{ij}$ represents the similarity between the $i$-th query and the $j$-th key.

### Step 2: Scale the Scores

To avoid extremely large values when $d_k$ is large, scale the dot products:

$$
\text{ScaledScores} = \frac{QK^T}{\sqrt{d_k}}
$$

### Step 3: Apply Softmax

Convert the scores to probabilities:

$$
\text{AttentionWeights} = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)
$$

Softmax is applied row-wise across each query's scores.

### Step 4: Weighted Sum of Values

Use the attention weights to combine the values:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

- Output shape: $\mathbb{R}^{n_q \times d_v}$
- Each query attends to all values with a learned focus.

---

## Multi-Head Attention (Brief Overview)

To allow the model to jointly attend to information from different representation subspaces:

1. Linearly project $Q, K, V$ multiple times (typically 8 or 12 heads).
2. Compute attention in parallel.
3. Concatenate and project the outputs:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

Where each head:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

---

## Backpropagation in Attention

Each operation (dot product, scaling, softmax, and matrix multiplication) is differentiable, so gradients can be propagated back through the attention computation during training.

- Derivatives are computed using the chain rule.
- Special care is taken when differentiating through **softmax**, which involves the **Jacobian matrix** of the softmax function.