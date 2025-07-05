from model import NeuralNetwork
import numpy as np

np.random.seed(42)

X = np.random.randn(100, 3)

y = (np.sum(X, axis=1) > 0).astype(float).reshape(-1, 1)

model = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1)
model.fit(X, y, epochs=100, batch_size=32, learning_rate=0.001)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(model.history)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()