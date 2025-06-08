import numpy as np
import re

class Vocabulary:
    def __init__(self, vocabulary_path: str) -> None:
        with open(vocabulary_path, "r") as f:
            self.sentences = [line.strip() for line in f.readlines()]
        self.size = 0

    def __clean_sentence(self, sentence: str) -> str:
        return re.sub(r'[^\w\s]', '', sentence.lower())

    def run(self):
        self.vocabulary = {
            "<eos>": 0,
            "<sos>": 1,
            "<pad>": 2,
            "<unk>": 3,
        }
        counter = len(self.vocabulary)

        for sentence in self.sentences:
            sentence = self.__clean_sentence(sentence)
            for word in sentence.split():
                if word not in self.vocabulary:
                    self.vocabulary[word] = counter
                    counter += 1

        self.size = counter
        return self.vocabulary
    
    def get_word_index(self, word: str) -> int:
        return self.vocabulary.get(word, self.vocabulary["<unk>"])

class InputEmbedding:
    def __init__(self, x, vocab_size, d_model, vocabulary):
        self.x = x
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.vocabulary = vocabulary
        self.embedding = np.random.randn(vocab_size, d_model)
        self.token_ids = self.__tokenizer(self.x)
        self.seq_length = len(self.token_ids)

    def __tokenizer(self, text):
        return [self.vocabulary.get_word_index(word) for word in text.split()]

    def __positional_embedding(self):
        PE = np.zeros((self.seq_length, self.d_model))
        position = np.arange(self.seq_length)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        PE[:, 0::2] = np.sin(position[:, np.newaxis] * div_term)
        PE[:, 1::2] = np.cos(position[:, np.newaxis] * div_term)
        
        return PE

    def forward(self):
        embedded = self.embedding[self.token_ids]
        pos_encoded = embedded + self.__positional_embedding()
        return pos_encoded

def softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    return np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=-1, keepdims=True)

class SelfAttention:
    def __init__(self, d_model=512, d_k=64):
        self.d_model = d_model
        self.d_k = d_k

    def forward(self, Q, K, V):
        num_term = Q @ K.T
        denom_term = np.sqrt(self.d_k)
        attention_scores = num_term / denom_term
        attention_weights = softmax(attention_scores)
        return attention_weights @ V

class MultiHeadSelfAttention:
    def __init__(self, d_model=512,h=8) -> None:
        self.h = h
        self.d_model = d_model
        self.d_k = self.d_model // self.h
        self.d_v = self.d_model // self.h
        self.Wq = [np.random.rand(self.d_model, self.d_k) for _ in range(self.h)]
        self.Wk = [np.random.rand(self.d_model, self.d_k) for _ in range(self.h)]
        self.Wv = [np.random.rand(self.d_model, self.d_v) for _ in range(self.h)]
        self.Wo = np.random.rand(self.d_v * self.h, self.d_model)

    def forward(self, X):
        Q = np.stack([X @ self.Wq[i] for i in range(self.h)])
        K = np.stack([X @ self.Wk[i] for i in range(self.h)])
        V = np.stack([X @ self.Wv[i] for i in range(self.h)])
        heads = [SelfAttention(d_model=self.d_model, d_k=self.d_k).forward(Q[i], K[i], V[i]) for i in range(self.h)]
        return np.concatenate(heads, axis=-1) @ self.Wo

# I hesitated between using Layer Normalization or AdaNorm as described in this paper: https://arxiv.org/pdf/1911.07013
# I decided to implement Layer Normalization as the original paper uses it. But it's good to know that AdaNorm solves 
# the overfitting problem of Layer Normalization because of the gamma and beta parameters.
# However, I wanted to stick to the original paper.
class LayerNormalization:
    def __init__(self, d_model=512, lr=0.01):
        self.lr = lr
        self.d_model = d_model
        self.epsilon = 1e-5
        self.gamma = np.ones((1, self.d_model))
        self.beta = np.zeros((1, self.d_model))

    def forward(self, X):
        mu = np.mean(X, axis=1, keepdims=True)
        sigma = np.std(X, axis=1, keepdims=True) + self.epsilon
        x_hat = (X - mu) / sigma
        return self.gamma * x_hat + self.beta

    def backprop(self, grad_output):
        grad_gamma = np.sum(grad_output * self.x_hat, axis=0)
        grad_beta = np.sum(grad_output, axis=0)
        return grad_gamma, grad_beta

    def update_parameters(self, grad_gamma, grad_beta):
        self.gamma -= self.lr * grad_gamma
        self.beta -= self.lr * grad_beta

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class FeedForward:
    def __init__(self, d_model=512, d_ff=2048, lr=0.01):
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = np.random.rand(self.d_model, self.d_ff)
        self.W2 = np.random.rand(self.d_ff, self.d_model)
        self.b1 = np.zeros((1, self.d_ff))
        self.b2 = np.zeros((1, self.d_model))

    def forward(self, X):
        self.X = X
        return np.maximum(0, X @ self.W1 + self.b1) @ self.W2 + self.b2
    
    def backprop(self, z1, z2, grad_output_1, grad_output_2):
        grad_z2 = grad_output_1
        grad_z1 = grad_output_2 @ relu_derivative(z1)
        grad_W2 = grad_z2 @ z1.T
        grad_b2 = np.sum(grad_z2, axis=0)
        grad_W1 = grad_z1 @ self.X.T
        grad_b1 = np.sum(grad_z1, axis=0)
        return grad_W1, grad_b1, grad_W2, grad_b2

    def update_parameters(self, grad_W1, grad_b1, grad_W2, grad_b2):
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2


class Encoder:
    def __init__(self, d_model=512, h=8, d_ff=2048):
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.Wq = np.random.rand(self.d_model, self.d_model)
        self.Wk = np.random.rand(self.d_model, self.d_model)

    def forward(self, X):
        # at this point we consider that the input is already embedded
        multi_head_attention = MultiHeadSelfAttention(d_model=self.d_model, h=self.h).forward(X)
        add_and_layer_norm = LayerNormalization(d_model=self.d_model).forward(multi_head_attention + X)
        feed_forward = FeedForward(d_model=self.d_model, d_ff=self.d_ff).forward(add_and_layer_norm)
        second_add_and_layer_norm = LayerNormalization(d_model=self.d_model).forward(feed_forward + add_and_layer_norm)
        return second_add_and_layer_norm

class Transformer:
    def __init__(self, d_model=512, h=8, d_ff=2048):
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.Wq = np.random.rand(self.d_model, self.d_model)
        self.Wk = np.random.rand(self.d_model, self.d_model)
        self.Wv = np.random.rand(self.d_model, self.d_model)


if __name__ == "__main__":
    english_vocabulary_path = "/home/amaury-delille/Documents/machine_learning/ml-dl/attention-is-all-you-need/datasets/en_sents"
    vocabulary = Vocabulary(english_vocabulary_path)
    english_vocabulary = vocabulary.run()
    x = "Be quiet for a moment."
    input_embedding = InputEmbedding(x=x, vocab_size=vocabulary.size, d_model=512, vocabulary=vocabulary)
    embeddings = input_embedding.forward()
    
    encoder = Encoder(d_model=512, h=8, d_ff=2048)
    output = encoder.forward(X=embeddings)
    print(output)