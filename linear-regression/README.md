# Mathematics Behind Linear Regression

## Problem Statement

Given a dataset with input vectors $$x \in \mathbb{R}^n$$ and corresponding scalar outputs $$y \in \mathbb{R}$$, we aim to learn a linear function:

$$
\hat{y} = w^T x + b
$$

Where:
- $w \in \mathbb{R}^n$ is the weight vector
- $b \in \mathbb{R}$ is the bias term
- $\hat{y}$ is the model's prediction

---

## Variables and Dimensions

| Symbol      | Description                    | Dimensions         |
|-------------|--------------------------------|--------------------|
| $$x$$       | Input feature vector           | $$(n \times 1)$$   |
| $$w$$       | Weight vector                  | $$(n \times 1)$$   |
| $$b$$       | Bias scalar                    | $$(1 \times 1)$$   |
| $$y$$       | True output (target)           | $$(1 \times 1)$$   |
| $$\hat{y}$$ | Predicted output               | $$(1 \times 1)$$   |
| $$m$$       | Number of training examples    | —                  |

In practice, for a dataset with $m$ samples, the full input matrix is:

- $$X \in \mathbb{R}^{m \times n}$$
- $$y \in \mathbb{R}^{m \times 1}$$
- $$\hat{y} = Xw + b$$

---

## Cost Function

We use **Mean Squared Error (MSE)** as the cost function:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^m ( \hat{y}^{(i)} - y^{(i)} )^2 = \frac{1}{m} \| Xw + b - y \|^2
$$

This measures the average squared difference between the predicted and actual values.

---

## Gradient Computation

We want to minimize the cost function $J(w, b)$ with respect to $w$ and $b$ using **gradient descent**.

### Gradients

Let:

$
\hat{y} = Xw + b
$

Then:

- Gradient w.r.t. $w$:
  $$
  \frac{\partial J}{\partial w} = \frac{2}{m} X^T (Xw + b - y)
  $$

- Gradient w.r.t. $b$:
  $$
  \frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) = \frac{2}{m} \cdot \mathbf{1}^T (Xw + b - y)
  $$

Where $\mathbf{1}$ is a column vector of ones of shape $(m \times 1)$.

---
## Gradient Descent

Given a learning rate $\eta > 0$, we update the parameters as:

### Update Equations:

$$
w := w - \eta \cdot \frac{\partial J}{\partial w}
$$

$$
b := b - \eta \cdot \frac{\partial J}{\partial b}
$$

Repeat until convergence (i.e., until cost decreases below a threshold or a fixed number of epochs is reached).
