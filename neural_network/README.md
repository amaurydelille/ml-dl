# Maths behind neural networks

## Variables

**Layers**
$a_0 = [x_1, ..., x_n]^T, n$ = number inputs 
$a_1 = [x_1, ..., x_n]^T, n$ = number hidden neurons 
$z = [x_1, ..., x_n]^T, n$ = number outputs 

**Weights and biases**
$w_1$ = weights matrix between layer 0 and 1
$w_2$ = weights matrix between layer 1 and output
$b_1$ = biases of hidden layer
$b_2$ = biases of output layer

**Functions**
$g(x)$ = sigmoid of x

$\hat{y}$ = prediction of our model

## 1. Forward propagation
$a_0$ has been fed by our dataset, we need to compute its weighted sum and to communicate it to $a_1$ and so on until we reach $z$.

$a_1 = g(\sum_{i=1}^{n}w_{1, i}x_i + b_1)$

$z = g(\sum_{i=1}^{n}w_{2, i}x_i + b_2)$

## 2. Back propagation
Until now, we were able to produce an output, we don't know if it's correct or not, a way to measure it would be to use a cost function, the more its result is close to 0, the less we computed errors, the more our model is accurate.

$C(\theta)=\frac{1}{n}\sum_{i=0}^{n}(y_i - \hat{y_i})^2$

Now we got an error, we want to minimize it $\implies$ find $\theta$ where $C=0 \implies$ compute how $C$ is sensitive to the change of $\theta$. How to achieve that ? We derivate, we find a gradient that indicates the direction and the speed of change of $C$ when $\theta$ varies.

$\frac{\partial{C}}{\partial{\theta}}= \Delta{C(\theta)}$

The thing is we can not only derivate $C$ because its the result of other nested function. So we use the chain rule calculs that states:
$z = f(g(h(x)))$
then 
$\frac{\partial{z}}{\partial{x}} = \frac{\partial{f}}{\partial{g}}\frac{\partial{g}}{\partial{h}}\frac{\partial{h}}{\partial{x}}$

