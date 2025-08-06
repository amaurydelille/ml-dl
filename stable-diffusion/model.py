from math import sqrt
import numpy as np
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)

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

    def forward_process(self, x: np.array) -> np.array:
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

if __name__ == "__main__":
    input_dim = 512
    x = Image.open("/Users/amaurydelille/Documents/deep-learning/ml-dl/stable-diffusion/stable-diffusion-face-dataset/512/man/man_0001.jpg")
    x = np.array(x)
    stable_diffusion = StableDiffusion(input_dim=input_dim, hidden_dim=128, output_dim=input_dim)
    x = stable_diffusion.forward_process(x)
    x = Image.fromarray((x * 255).astype("uint8"))
    x.show()