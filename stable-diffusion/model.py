from math import sqrt
import numpy as np
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)

def siLU(x: np.array) -> np.array:
    return x * (x > 0) + 0.01 * x * (x <= 0)

def reLU(x: np.array) -> np.array:
    return np.maximum(0, x)

def reLU_derivative(x: np.array) -> np.array:
    return np.where(x > 0, 1, 0)

def loss(y_pred: np.array, y: np.array) -> float:
    return np.mean(np.abs(y_pred - y))

class UNet:
    def __init__(self, input_dim: int, timestep: int) -> None:
        self.input_dim = input_dim
        self.timestep = timestep

    class ResNetEncoder:
        def __init__() -> None:
            pass

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
                

        class GroupNormalization:
            def __init__(self, x: np.array, G: int) -> None:
                self.G = G
                self.x = x
                self.gamma = np.random.rand((1, G))
                self.beta = np.random.rand((1, G))

            def forward(self, x: np.array) -> np.array:
                B, C, H, W = x.shape
                xg = x.reshape(B, self.G, C // self.G, H, W)
                mean_g = xg.mean(axis=(2, 3, 4), keepdims=True)
                var_g = xg.var(axis=(2, 3, 4), keepdims=True)
                xg = (xg - mean_g) / np.sqrt(var_g + 1e-5)
                xg = xg.reshape(B, C, H, W)
                self.gamma = np.ones((1, C, 1, 1))
                self.beta = np.zeros((1, C, 1, 1))

                return xg * self.gamma + self.beta
            
        def forward(self, x: np.array) -> np.array:
            gn = self.GroupNormalization(x, self.G)
            sliu_gn = siLU(gn)

class StableDiffusion:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        logging.info(f"Initializing StableDiffusion with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.beta_start = 0.0001
        self.beta_end = 0.02

        self.steps = 1000
        self.betas = np.linspace(self.beta_start, self.beta_end, self.steps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1 - self.alpha_cumprod)

    class ForwardProcess:
        def run(self, x: np.array) -> np.array:
            logging.info(f"Forward process with input shape {x.shape}")
            assert x.shape == (self.input_dim, self.input_dim, 3)

            # We need a forward process that would preserve the variance and converge
            # to a standard normal distribution. We would then need to affect the mean at
            # each step to make it converge to 0.
            x0 = x
            # epsilon is sampled from N(0, 1) because we need to ensure the noise is centered around zero.
            epsilon = np.random.normal(0, 1, size=x.shape)
            
            for i in range(self.steps):
                x = (
                    self.sqrt_alpha_cumprod[i] * x0
                    + self.sqrt_one_minus_alpha_cumprod[i] * epsilon
                )

            return x

    class ReverseProcess:
        def __forward(self, x: np.array) -> np.array:
            self.a1 



if __name__ == "__main__":
    input_dim = 512
    x = Image.open("/Users/amaurydelille/Documents/deep-learning/ml-dl/stable-diffusion/stable-diffusion-face-dataset/512/man/man_0001.jpg")
    x = np.array(x)
    stable_diffusion = StableDiffusion(input_dim=input_dim, hidden_dim=128, output_dim=input_dim)
    x = stable_diffusion.ForwardProcess().run(x)
    x = Image.fromarray((x * 255).astype("uint8"))
    x.show()