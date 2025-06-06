# Neural Network Mathematics

## Architecture and Notation

We consider a feedforward neural network with the following components:

- Input layer of size $n$  
- Hidden layer of size $h$
- Output layer of size $m$

### Variables and Dimensions

| Symbol       | Description                            | Dimensions               |
|--------------|----------------------------------------|---------------------------|
| $$a_0$$      | Input vector                           | $$(n \times 1)$$          |
| $$w_1$$      | Weights (input → hidden)               | $$(h \times n)$$          |
| $$b_1$$      | Bias vector for hidden layer           | $$(h \times 1)$$          |
| $$a_1$$      | Activations of hidden layer            | $$(h \times 1)$$          |
| $$w_2$$      | Weights (hidden → output)              | $$(m \times h)$$          |
| $$b_2$$      | Bias vector for output layer           | $$(m \times 1)$$          |
| $$z$$        | Output vector (before activation)      | $$(m \times 1)$$          |
| $$\hat{y}$$  | Final output prediction                | $$(m \times 1)$$          |
| $$y$$        | Ground truth label                     | $$(m \times 1)$$          |

### Activation Function

We use the **sigmoid** activation function:

$$
g(x) = \frac{1}{1 + e^{-x}} \quad , \quad g'(x) = g(x)(1 - g(x))
$$

---

## 1. Forward Propagation

$$
z_1 = w_1 a_0 + b_1 \quad \text{(shape: } h \times 1 \text{)}
$$

$$
a_1 = g(z_1)
$$

$$
z_2 = w_2 a_1 + b_2 \quad \text{(shape: } m \times 1 \text{)}
$$

$$
\hat{y} = g(z_2)
$$

---

## 2. Backward Propagation

We use **Mean Squared Error (MSE)**:

$$
C = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

Let:

$$
\delta_2 = \frac{\partial C}{\partial z_2}
$$

Using the chain rule:

$$
\delta_2 = (\hat{y} - y) \circ g'(z_2)
$$

Where $\circ$ denotes element-wise multiplication.

Using the chain rule again:

$$
\delta_1 = (w_2^T \delta_2) \circ g'(z_1)
$$

---

## 3. Gradients of the Parameters

### Output Layer

- Gradient w.r.t weights:
  $$
  \frac{\partial C}{\partial w_2} = \delta_2 a_1^T
  $$

- Gradient w.r.t biases:
  $$
  \frac{\partial C}{\partial b_2} = \delta_2
  $$

### Hidden Layer

- Gradient w.r.t weights:
  $$
  \frac{\partial C}{\partial w_1} = \delta_1 a_0^T
  $$

- Gradient w.r.t biases:
  $$
  \frac{\partial C}{\partial b_1} = \delta_1
  $$

---

## 4. Parameter Update (Gradient Descent)

Given a learning rate $\eta$, update the parameters as follows:

### Update Equations:

$$
w_1 := w_1 - \eta \cdot \frac{\partial C}{\partial w_1}
$$

$$
b_1 := b_1 - \eta \cdot \frac{\partial C}{\partial b_1}
$$

$$
w_2 := w_2 - \eta \cdot \frac{\partial C}{\partial w_2}
$$

$$
b_2 := b_2 - \eta \cdot \frac{\partial C}{\partial b_2}
$$

Repeat forward and backward propagation over multiple epochs until convergence.