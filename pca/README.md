# Principal Component Analysis (PCA) - Mathematical Foundation

## Overview

Principal Component Analysis is a linear dimensionality reduction technique that transforms data to a lower-dimensional space while preserving maximum variance. It finds orthogonal directions of maximum variance in high-dimensional data.

## Mathematical Foundation

### Data Matrix

Given a dataset with $n$ observations and $p$ features:

$$
X = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}
$$

Where each row represents an observation and each column represents a feature.

### Data Centering

The first step is to center the data by subtracting the mean from each feature:

$$
\bar{x}_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}
$$

The centered data matrix becomes:

$$
X_c = X - \mathbf{1}\bar{X}^T
$$

Where $\mathbf{1}$ is a column vector of ones and $\bar{X}$ is the vector of column means.

### Covariance Matrix

The sample covariance matrix is computed as:

$$
C = \frac{1}{n-1} X_c^T X_c
$$

This is a p × p symmetric matrix where element $C_{ij}$ represents the covariance between features i and j:

$$
C_{ij} = \frac{1}{n-1} \sum_{k=1}^{n} (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)
$$

### Eigenvalue Decomposition

PCA finds the eigenvalues and eigenvectors of the covariance matrix:

$$
C \mathbf{v}_i = \lambda_i \mathbf{v}_i
$$

Where:
- $\lambda_i$ are eigenvalues
- $\mathbf{v}_i$ are corresponding eigenvectors (principal components)

The eigenvalues are ordered: $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$

### Principal Components

The principal components are the eigenvectors of the covariance matrix. The i-th principal component is:

$$
\mathbf{v}_i = \begin{bmatrix}
v_{i1} \\
v_{i2} \\
\vdots \\
v_{ip}
\end{bmatrix}
$$

These vectors are orthonormal: $\mathbf{v}_i^T \mathbf{v}_j = \delta_{ij}$

### Transformation

The data is projected onto the principal components:

$$
Y = X_c V
$$

Where V is the matrix of eigenvectors:

$$
V = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_p]
$$

### Dimensionality Reduction

To reduce dimensionality to k dimensions (k < p), select the first k principal components:

$$
V_k = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k]
$$

The reduced representation is:

$$
Y_k = X_c V_k
$$

### Variance Explained

The variance explained by the i-th principal component is:

$$
\text{Var}(\mathbf{v}_i) = \lambda_i
$$

The proportion of variance explained by the i-th component:

$$
\frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}
$$

The cumulative proportion of variance explained by the first k components:

$$
\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{p} \lambda_j}
$$

### Reconstruction

The original data can be approximated using k principal components:

$$
X_{rec} = Y_k V_k^T + \mathbf{1}\bar{X}^T
$$

### Reconstruction Error

The mean squared reconstruction error is:

$$
\text{MSE} = \frac{1}{np} \sum_{i=k+1}^{p} \lambda_i
$$

## Optimization Perspective

### Variance Maximization

PCA can be formulated as finding the direction that maximizes variance. For the first principal component:

$$
\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1} \mathbf{v}^T C \mathbf{v}
$$

This is solved using Lagrange multipliers:

$$
L = \mathbf{v}^T C \mathbf{v} - \lambda(\mathbf{v}^T \mathbf{v} - 1)
$$

Taking the derivative with respect to $\mathbf{v}$ and setting to zero:

$$
\frac{\partial L}{\partial \mathbf{v}} = 2C\mathbf{v} - 2\lambda\mathbf{v} = 0
$$

This gives us the eigenvalue equation: $C\mathbf{v} = \lambda\mathbf{v}$

### Sequential Optimization

Subsequent principal components are found by maximizing variance subject to orthogonality constraints:

$$
\mathbf{v}_k = \arg\max_{\|\mathbf{v}\|=1, \mathbf{v}^T\mathbf{v}_i=0 \text{ for } i<k} \mathbf{v}^T C \mathbf{v}
$$

## Singular Value Decomposition (SVD) Approach

Alternatively, PCA can be computed using SVD of the centered data matrix:

$$
X_c = U \Sigma V^T
$$

Where:
- U is an n × n orthogonal matrix
- $\Sigma$ is an n × p diagonal matrix with singular values
- V is a p × p orthogonal matrix

The relationship to eigendecomposition:
- Principal components: columns of V
- Eigenvalues: $\lambda_i = \frac{\sigma_i^2}{n-1}$

## Mathematical Properties

### Total Variance Conservation

The sum of all eigenvalues equals the trace of the covariance matrix:

$$
\sum_{i=1}^{p} \lambda_i = \text{tr}(C) = \sum_{i=1}^{p} C_{ii}
$$

### Orthogonality

Principal components are orthogonal:

$$
\mathbf{v}_i^T \mathbf{v}_j = 0 \text{ for } i \neq j
$$

### Optimality

PCA provides the optimal linear transformation for:
1. Maximum variance preservation
2. Minimum reconstruction error (in least squares sense)

### Invariance Properties

PCA is invariant under:
- Translation of data
- Orthogonal transformations

But not invariant under:
- Scaling of individual features
- General linear transformations

