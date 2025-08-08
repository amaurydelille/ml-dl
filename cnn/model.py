import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def reLU(x: np.array) -> np.array:
    return np.maximum(0, x)

def reLU_derivative(x: np.array) -> np.array:
    return np.where(x > 0, 1, 0)

def loss(y_pred: np.array, y: np.array) -> float:
    return np.mean(np.abs(y_pred - y))

class ConvolutionalNeuralNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, kernel_dim: int = 3, max_pooling_dim: int = 2, epochs: int = 100, learning_rate: float = 0.01) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kernel_dim = kernel_dim
        self.max_pooling_dim = max_pooling_dim
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.kernel = np.random.randn(kernel_dim, kernel_dim)
        self.max_pool = np.random.randn(max_pooling_dim, max_pooling_dim)
        self.features_map_dim = self.input_dim - self.kernel_dim + 1
        self.features_map = np.zeros((self.features_map_dim, self.features_map_dim))

        self.pool_out_h = max(1, self.features_map_dim // self.max_pooling_dim)
        self.pool_out_w = max(1, self.features_map_dim // self.max_pooling_dim)
        self.flattened_input_dim = self.pool_out_h * self.pool_out_w

        self.weights = {
            1: np.random.randn(self.flattened_input_dim, hidden_dim),
            2: np.random.randn(hidden_dim, output_dim)
        }

        self.biases = {
            1: np.zeros((1, hidden_dim)),
            2: np.zeros((1, output_dim))
        }

        logging.info(f"Kernel Dimension: {self.kernel_dim}")
        logging.info(f"Max Pooling Dimension: {self.max_pooling_dim}")
        logging.info(f"Features Map Dimension: {self.features_map_dim}")
        logging.info(f"Pooled Output Dimension: ({self.pool_out_h}, {self.pool_out_w}) -> flattened {self.flattened_input_dim}")
        logging.info(f"Weights Dimension: {self.weights[1].shape}, {self.weights[2].shape}")
        logging.info(f"Biases Dimension: {self.biases[1].shape}, {self.biases[2].shape}")

    def __convolution(self, x: np.array) -> np.array:
        for i in range(self.features_map_dim):
            for j in range(self.features_map_dim):
                window = x[i:i+self.kernel_dim, j:j+self.kernel_dim]
                self.features_map[i, j] = np.sum(window * self.kernel)

        stride = self.max_pooling_dim
        pooled = np.zeros((self.pool_out_h, self.pool_out_w))
        for i in range(self.pool_out_h):
            for j in range(self.pool_out_w):
                hs, ws = i * stride, j * stride
                pooled[i, j] = np.max(self.features_map[hs:hs+stride, ws:ws+stride])

        return pooled


    def forward(self, x: np.array) -> np.array:
        x = self.__convolution(x)
        if x.ndim == 2:
            x = x.reshape(1, -1)
        self.fc_input = x
        self.a1 = reLU(self.fc_input @ self.weights[1] + self.biases[1])
        a2 = reLU(self.a1 @ self.weights[2] + self.biases[2])
        return a2

    def backward(self, x: np.array, y_pred: np.array, y: np.array) -> None:
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if y.shape[0] != y_pred.shape[0]:
            y = y[:y_pred.shape[0], :]

        delta2 = y_pred - y
        delta1 = (delta2 @ self.weights[2].T) * reLU_derivative(self.a1)

        self.weights[2] -= self.learning_rate * (self.a1.T @ delta2)
        self.biases[2] -= self.learning_rate * delta2

        self.weights[1] -= self.learning_rate * (self.fc_input.T @ delta1)
        self.biases[1] -= self.learning_rate * delta1
        
    def fit(self, x: np.array, y: np.array) -> None:
        for i in range(self.epochs):
            y_pred = self.forward(x)
            y_target = y
            if y_target.ndim == 1:
                y_target = y_target.reshape(1, -1)
            if y_target.shape[0] != y_pred.shape[0]:
                y_target = y_target[:y_pred.shape[0], :]

            self.backward(x, y_pred, y_target)
            logging.info(f"Epoch {i+1}/{self.epochs}, Loss: {loss(y_pred, y_target)}")

    def predict(self, x: np.array) -> np.array:
        return self.forward(x)

    def evaluate(self, x: np.array, y: np.array) -> float:
        y_pred = self.predict(x)
        return loss(y_pred, y)

if __name__ == "__main__":
    x = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

    model = ConvolutionalNeuralNetwork(input_dim=4, hidden_dim=2, output_dim=2)
    model.fit(x, y)
    print(model.predict(x))
    print(model.evaluate(x, y))