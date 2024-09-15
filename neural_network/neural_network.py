import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ---------------------------------------------------------------
# ---------------------------- utils ----------------------------
# ---------------------------------------------------------------
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def reLU(x):
    return np.maximum(x, np.zeros(x.shape))
    # return np.array([max(0, xi) for xi in x]).reshape(-1, 1)

def reLU_derivative(x):
    return (x > 0).astype(float)

def MSE(y, y_pred) -> float:
    return np.square(y - y_pred).mean()

def binary_cross_entropy(y: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y)
    return (-1/n) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def compute_loss(method: str, y: np.ndarray, y_pred: np.ndarray) -> float:
    if method == 'mse':
        return np.square(y - y_pred).mean()
    elif method == 'bce':
        return (-1/len(y)) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    
def R2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    
    return r2


# ---------------------------------------------------------------        
# ------------------------ NeuralNetwork ------------------------
# ---------------------------------------------------------------
matplotlib.use('TkAgg')
class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def init_params(self):
        w1 = np.random.standard_normal(size=(self.hidden_size, self.input_size)) * np.sqrt(1. / (self.input_size * self.hidden_size))
        b1 = np.random.standard_normal(size=(self.hidden_size, 1)) / np.sqrt(self.hidden_size)
        w2 = np.random.standard_normal(size=(self.output_size, self.hidden_size)) * np.sqrt(1. / self.hidden_size)
        b2 = np.random.standard_normal(size=(self.output_size, 1))

        return w1, b1, w2, b2

    def forward_propagation(self, input: np.ndarray, w1, b1, w2, b2):
        # input shape (batch size, input size)
        self.hidden = w1.dot(input.T) + b1  # (hidden size, batch size) 
        self.ckpt1 = self.hidden.copy()
        self.hidden = reLU(self.hidden)
        self.output = w2.dot(self.hidden) + b2  # (output size, batch size)
        return self.output.flatten()

    def back_propagation(self, input: np.ndarray, y: np.ndarray, w2):
        dcdy = 1/y.size * (self.output.flatten() - y).reshape(-1, 1) # (batch size,)
        # print("hidden: ", self.hidden)
        dw2 = (self.hidden @ dcdy).T
        db2 = dcdy.copy()  # (batch size, 1)
        dz1 = (w2.T @ dcdy.T) * reLU_derivative(self.ckpt1)  # (hidden size, batch size)
        dw1 = dz1.dot(input)  # (hidden size, input size)
        db1 = dz1.copy()  # (hidden size, batch size)
        return dw1, db1.mean(axis=1).reshape(-1, 1), dw2, db2.mean(axis=0)

    def update_weights(self, dw1, db1, dw2, db2, w1, b1, w2, b2):

        w1 -= self.learning_rate * dw1
        b1 -= self.learning_rate * db1
        w2 -= self.learning_rate * dw2
        b2 -= self.learning_rate * db2

        return w1, b1, w2, b2

    def gradient_descent(self, x: np.ndarray, y: np.ndarray):
        # x shape (batch size, features)
        loss_history = []
        w1, b1, w2, b2 = self.init_params()
        batch_size = 512
        for epoch in range(self.epochs):
            for i in range(0, x.shape[0], batch_size):
                x_input = x[i:i+batch_size]
                y_true = y[i:i+batch_size]
                output = self.forward_propagation(x_input, w1, b1, w2, b2)
                dw1, db1, dw2, db2 = self.back_propagation(x_input, y_true, w2)
                w1, b1, w2, b2 = self.update_weights(dw1=dw1, db1=db1, dw2=dw2, db2=db2, w1=w1, b1=b1, w2=w2, b2=b2)
                loss = compute_loss('mse', y_true, output)
                loss_history.append(loss)

        return w1, b1, w2, b2, loss_history

    def predict(self, x: np.ndarray, y, w1, b1, w2, b2):
        pred = self.forward_propagation(x, w1, b1, w2, b2)

        out = np.vstack([
            pred,
            y,
            1/2 * (pred - y)**2
        ])

        loss = MSE(y, pred)
        return pred, loss
    
    def get_accuracy(self, y_true, y_pred):
        return R2_score(y_true, y_pred)

    def display_learning_curve(self, loss_history):
        plt.plot(range(1, len(loss_history) + 1), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.grid(True)
        plt.show()
