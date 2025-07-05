import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))

        self.m_w1 = np.zeros_like(self.W1)
        self.v_w1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_w2 = np.zeros_like(self.W2)
        self.v_w2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        self.t = 0

        self.history = []

    def __leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.maximum(alpha * x, x)
    
    def __leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)

    def forward(self, X: np.ndarray) -> None:
        self.X = X
        self.a1 = self.W1 @ X.T + self.b1  # shape: (hidden_dim, batch_size)
        self.h = self.__leaky_relu(self.a1)  # shape: (hidden_dim, batch_size)
        self.z = self.W2 @ self.h + self.b2  # shape: (output_dim, batch_size)

    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        batch_size = X.shape[0]
        
        # y shape: (batch_size, output_dim), y_pred shape: (batch_size, output_dim)
        delta_2 = -(y - y_pred).T  # shape: (output_dim, batch_size)
        self.dldw2 = delta_2 @ self.h.T / batch_size  # shape: (output_dim, hidden_dim)
        self.dldb2 = np.mean(delta_2, axis=1, keepdims=True)  # shape: (output_dim, 1)
        
        delta_1 = (self.W2.T @ delta_2) * self.__leaky_relu_derivative(self.a1)  # shape: (hidden_dim, batch_size)
        self.dldw1 = delta_1 @ X / batch_size  # shape: (hidden_dim, input_dim)
        self.dldb1 = np.mean(delta_1, axis=1, keepdims=True)  # shape: (hidden_dim, 1)

    def adam_optimizer(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.t += 1

        self.m_w1 = beta_1 * self.m_w1 + (1 - beta_1) * self.dldw1
        self.v_w1 = beta_2 * self.v_w1 + (1 - beta_2) * np.square(self.dldw1)
        m_w1_hat = self.m_w1 / (1 - beta_1**self.t)
        v_w1_hat = self.v_w1 / (1 - beta_2**self.t)
        self.W1 -= learning_rate * m_w1_hat / (np.sqrt(v_w1_hat) + epsilon)

        self.m_b1 = beta_1 * self.m_b1 + (1 - beta_1) * self.dldb1
        self.v_b1 = beta_2 * self.v_b1 + (1 - beta_2) * np.square(self.dldb1)
        m_b1_hat = self.m_b1 / (1 - beta_1**self.t)
        v_b1_hat = self.v_b1 / (1 - beta_2**self.t)
        self.b1 -= learning_rate * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)

        self.m_w2 = beta_1 * self.m_w2 + (1 - beta_1) * self.dldw2
        self.v_w2 = beta_2 * self.v_w2 + (1 - beta_2) * np.square(self.dldw2)
        m_w2_hat = self.m_w2 / (1 - beta_1**self.t)
        v_w2_hat = self.v_w2 / (1 - beta_2**self.t)
        self.W2 -= learning_rate * m_w2_hat / (np.sqrt(v_w2_hat) + epsilon)

        self.m_b2 = beta_1 * self.m_b2 + (1 - beta_1) * self.dldb2
        self.v_b2 = beta_2 * self.v_b2 + (1 - beta_2) * np.square(self.dldb2)
        m_b2_hat = self.m_b2 / (1 - beta_1**self.t)
        v_b2_hat = self.v_b2 / (1 - beta_2**self.t)
        self.b2 -= learning_rate * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 32, learning_rate: float = 0.001) -> None:
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            total_loss = 0
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                self.forward(X_batch)
                y_pred = self.z.T

                loss = np.mean(np.square(y_batch - y_pred))
                total_loss += loss

                self.backward(X_batch, y_batch, y_pred)
                self.adam_optimizer(learning_rate=learning_rate)

            avg_loss = total_loss / n_batches
            self.history.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

        