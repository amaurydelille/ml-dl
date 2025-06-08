import numpy as np
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Vocabulary:
    def __init__(self, vocabulary_path: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing vocabulary from {vocabulary_path}")
        with open(vocabulary_path, "r") as f:
            self.sentences = [line.strip() for line in f.readlines()]
        self.size = 0

    def __clean_sentence(self, sentence: str) -> str:
        return re.sub(r'[^\w\s]', '', sentence.lower())

    def run(self):
        self.logger.info("Building vocabulary")
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
        self.logger.info(f"Vocabulary built with size {self.size}")
        return self.vocabulary
    
    def get_word_index(self, word: str) -> int:
        return self.vocabulary.get(word, self.vocabulary["<unk>"])

class InputEmbedding:
    def __init__(self, x, vocab_size, d_model, vocabulary):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing input embedding with vocab_size={vocab_size}, d_model={d_model}")
        self.x = x
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.vocabulary = vocabulary
        self.embedding = np.random.randn(vocab_size, d_model)
        self.max_seq_length = max(len(sentence.split()) for sentence in self.x)
        self.batch_size = len(self.x)
        self.token_ids = self.__tokenizer(self.x)
        self.logger.info(f"Batch size: {self.batch_size}, Max sequence length: {self.max_seq_length}")

    def __tokenizer(self, text):
        self.logger.info("Tokenizing input text with padding")
        batch_tokens = []
        for sentence in text:
            words = sentence.split()
            tokens = [self.vocabulary.get_word_index(word) for word in words]
            padding_length = self.max_seq_length - len(tokens)
            if padding_length > 0:
                tokens.extend([self.vocabulary.get_word_index("<pad>")] * padding_length)
            batch_tokens.append(tokens)
        return np.array(batch_tokens)

    def __positional_embedding(self):
        self.logger.info("Computing positional embeddings")
        PE = np.zeros((self.max_seq_length, self.d_model))
        position = np.arange(self.max_seq_length)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        PE[:, 0::2] = np.sin(position[:, np.newaxis] * div_term)
        PE[:, 1::2] = np.cos(position[:, np.newaxis] * div_term)
        
        return PE

    def forward(self):
        self.logger.info("Forward pass: combining word embeddings with positional encodings")
        embedded = self.embedding[self.token_ids]  # shape: (batch_size, max_seq_length, d_model)
        pos_encoded = embedded + self.__positional_embedding()[np.newaxis, :, :]
        return pos_encoded

def softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    return np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=-1, keepdims=True)

class SelfAttention:
    def __init__(self, d_model=512, d_k=64):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing self attention with d_model={d_model}, d_k={d_k}")
        self.d_model = d_model
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        self.logger.info("Computing self attention forward pass")
        seq_len = Q.shape[0]
        
        num_term = Q @ K.T
        denom_term = np.sqrt(self.d_k)
        attention_scores = num_term / denom_term
        
        if mask is not None:
            self.logger.info("Applying attention mask")
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            attention_scores = np.where(mask == 0, attention_scores, -1e9)
            
        attention_weights = softmax(attention_scores)
        self.logger.info("Self attention computation completed")
        return attention_weights @ V

class MultiHeadSelfAttention:
    def __init__(self, d_model=512, h=8) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing multi-head attention with d_model={d_model}, heads={h}")
        self.h = h
        self.d_model = d_model
        self.d_k = self.d_model // self.h
        self.d_v = self.d_model // self.h
        self.Wq = [np.random.randn(self.d_model, self.d_k) for _ in range(self.h)]
        self.Wk = [np.random.randn(self.d_model, self.d_k) for _ in range(self.h)]
        self.Wv = [np.random.randn(self.d_model, self.d_v) for _ in range(self.h)]
        self.Wo = np.random.randn(self.d_v * self.h, self.d_model)

    def forward(self, X, mask=None, key_value_states=None):
        self.logger.info("Starting multi-head attention forward pass")
        if key_value_states is None:
            self.logger.info("Computing self attention")
            Q = np.stack([X @ self.Wq[i] for i in range(self.h)])
            K = np.stack([X @ self.Wk[i] for i in range(self.h)])
            V = np.stack([X @ self.Wv[i] for i in range(self.h)])
        else:
            self.logger.info("Computing cross attention")
            Q = np.stack([X @ self.Wq[i] for i in range(self.h)])
            K = np.stack([key_value_states @ self.Wk[i] for i in range(self.h)])
            V = np.stack([key_value_states @ self.Wv[i] for i in range(self.h)])
        
        heads = [SelfAttention(d_model=self.d_model, d_k=self.d_k).forward(Q[i], K[i], V[i], mask) for i in range(self.h)]
        self.logger.info("Multi-head attention computation completed")
        return np.concatenate(heads, axis=-1) @ self.Wo

# I hesitated between using Layer Normalization or AdaNorm as proposed in this paper: https://arxiv.org/pdf/1911.07013
# I decided to implement Layer Normalization as the original paper uses it. But it's good to know that AdaNorm solves 
# the overfitting problem of Layer Normalization because of the gamma and beta parameters.
# However, I wanted to stick to the original paper.
class LayerNormalization:
    def __init__(self, d_model=512, lr=0.01):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing layer normalization with d_model={d_model}, lr={lr}")
        self.lr = lr
        self.d_model = d_model
        self.epsilon = 1e-5
        self.gamma = np.ones((1, self.d_model))
        self.beta = np.zeros((1, self.d_model))

    def forward(self, X):
        self.logger.info("Computing layer normalization forward pass")
        mu = np.mean(X, axis=1, keepdims=True)
        sigma = np.std(X, axis=1, keepdims=True) + self.epsilon
        x_hat = (X - mu) / sigma
        return self.gamma * x_hat + self.beta

    def backprop(self, grad_output):
        self.logger.info("Computing layer normalization backpropagation")
        grad_gamma = np.sum(grad_output * self.x_hat, axis=0)
        grad_beta = np.sum(grad_output, axis=0)
        return grad_gamma, grad_beta

    def update_parameters(self, grad_gamma, grad_beta):
        self.logger.info("Updating layer normalization parameters")
        self.gamma -= self.lr * grad_gamma
        self.beta -= self.lr * grad_beta

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class FeedForward:
    def __init__(self, d_model=512, d_ff=2048, lr=0.01):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing feed forward network with d_model={d_model}, d_ff={d_ff}, lr={lr}")
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = np.random.rand(self.d_model, self.d_ff)
        self.W2 = np.random.rand(self.d_ff, self.d_model)
        self.b1 = np.zeros((1, self.d_ff))
        self.b2 = np.zeros((1, self.d_model))

    def forward(self, X):
        self.logger.info("Computing feed forward network forward pass")
        self.X = X
        return np.maximum(0, X @ self.W1 + self.b1) @ self.W2 + self.b2
    
    def backprop(self, z1, z2, grad_output_1, grad_output_2):
        self.logger.info("Computing feed forward network backpropagation")
        grad_z2 = grad_output_1
        grad_z1 = grad_output_2 @ relu_derivative(z1)
        grad_W2 = grad_z2 @ z1.T
        grad_b2 = np.sum(grad_z2, axis=0)
        grad_W1 = grad_z1 @ self.X.T
        grad_b1 = np.sum(grad_z1, axis=0)
        return grad_W1, grad_b1, grad_W2, grad_b2

    def update_parameters(self, grad_W1, grad_b1, grad_W2, grad_b2):
        self.logger.info("Updating feed forward network parameters")
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2

class DropoutLayer:
    def __init__(self, dropout_rate=0.1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing dropout layer with rate={dropout_rate}")
        self.dropout_rate = dropout_rate

    def forward(self, X):
        self.logger.info("Applying dropout")
        self.mask = np.random.rand(*X.shape) > self.dropout_rate
        return X * self.mask

class Encoder:
    def __init__(self, d_model=512, h=8, d_ff=2048):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing encoder with d_model={d_model}, h={h}, d_ff={d_ff}")
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.Wq = np.random.rand(self.d_model, self.d_model)
        self.Wk = np.random.rand(self.d_model, self.d_model)

    def forward(self, X):
        self.logger.info("Starting encoder forward pass")
        self.logger.info("Computing multi-head self attention")
        multi_head_attention = MultiHeadSelfAttention(d_model=self.d_model, h=self.h).forward(X)
        self.logger.info("Applying add & norm after attention")
        add_and_layer_norm = LayerNormalization(d_model=self.d_model).forward(multi_head_attention + X)
        self.logger.info("Applying dropout after attention")
        sub_layer_1 = DropoutLayer(dropout_rate=0.1).forward(add_and_layer_norm)
        
        self.logger.info("Computing feed forward")
        feed_forward = FeedForward(d_model=self.d_model, d_ff=self.d_ff).forward(sub_layer_1)
        self.logger.info("Applying add & norm after feed forward")
        second_add_and_layer_norm = LayerNormalization(d_model=self.d_model).forward(feed_forward + sub_layer_1)
        self.logger.info("Applying dropout after feed forward")
        sub_layer_2 = DropoutLayer(dropout_rate=0.1).forward(second_add_and_layer_norm)
        self.logger.info("Encoder forward pass completed")
        return sub_layer_2
    
class LinearLayer:
    def __init__(self, d_model=512, d_out=512, lr=0.01):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing linear layer with d_model={d_model}, d_out={d_out}, lr={lr}")
        self.d_model = d_model
        self.d_out = d_out
        self.W = np.random.rand(self.d_model, self.d_out)
        self.b = np.zeros((1, self.d_out))
        self.lr = lr

    def forward(self, X):
        self.logger.info("Computing linear layer forward pass")
        return X @ self.W + self.b
    
    def backprop(self, X, grad_output):
        self.logger.info("Computing linear layer backpropagation")
        grad_W = grad_output.T @ X
        grad_b = np.sum(grad_output, axis=0)
        return grad_W, grad_b
    
    def update_parameters(self, grad_W, grad_b):
        self.logger.info("Updating linear layer parameters")
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b

def loss_function(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class Decoder:
    def __init__(self, d_model=512, h=8, d_ff=2048):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing decoder with d_model={d_model}, h={h}, d_ff={d_ff}")
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.masked_self_attention = MultiHeadSelfAttention(d_model=self.d_model, h=self.h)
        self.cross_attention = MultiHeadSelfAttention(d_model=self.d_model, h=self.h)
        self.feed_forward = FeedForward(d_model=self.d_model, d_ff=self.d_ff)

    def forward(self, X_enc, X_dec):
        self.logger.info("Starting decoder forward pass")
        self.logger.info("Computing masked multi-head self attention")
        multi_head_attention = self.masked_self_attention.forward(X_dec, mask=True)
        self.logger.info("Applying add & norm after masked attention")
        add_and_layer_norm = LayerNormalization(d_model=self.d_model).forward(multi_head_attention + X_dec)
        self.logger.info("Applying dropout after masked attention")
        sub_layer_1 = DropoutLayer(dropout_rate=0.1).forward(add_and_layer_norm)
        
        self.logger.info("Computing cross attention")
        cross_attention = self.cross_attention.forward(sub_layer_1, key_value_states=X_enc)
        self.logger.info("Applying add & norm after cross attention")
        add_and_layer_norm = LayerNormalization(d_model=self.d_model).forward(cross_attention + sub_layer_1)       
        self.logger.info("Applying dropout after cross attention")
        sub_layer_2 = DropoutLayer(dropout_rate=0.1).forward(add_and_layer_norm)
        
        self.logger.info("Computing feed forward")
        feed_forward = self.feed_forward.forward(sub_layer_2)
        self.logger.info("Applying add & norm after feed forward")
        third_add_and_layer_norm = LayerNormalization(d_model=self.d_model).forward(feed_forward + sub_layer_2)
        self.logger.info("Applying dropout after feed forward")
        sub_layer_3 = DropoutLayer(dropout_rate=0.1).forward(third_add_and_layer_norm)
        self.logger.info("Decoder forward pass completed")
        return sub_layer_3
    
class Transformer:
    def __init__(self, d_model=512, h=8, d_ff=2048, lr=0.01, epochs=100):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing transformer with d_model={d_model}, h={h}, d_ff={d_ff}, lr={lr}, epochs={epochs}")
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.encoder = Encoder(d_model=self.d_model, h=self.h, d_ff=self.d_ff)
        self.decoder = Decoder(d_model=self.d_model, h=self.h, d_ff=self.d_ff)
        self.linear_layer = LinearLayer(d_model=self.d_model, d_out=self.d_model, lr=0.01)
        self.lr = lr
        self.epochs = epochs

    def forward(self, X_enc, X_dec):
        self.logger.info("Starting transformer forward pass")
        self.logger.info("Encoding input sequence")
        X_enc = self.encoder.forward(X_enc)
        self.logger.info("Decoding with encoded sequence")
        X_dec = self.decoder.forward(X_enc, X_dec)
        self.logger.info("Applying final linear transformation")
        X_dec = self.linear_layer.forward(X_dec)
        self.logger.info("Computing final softmax")
        predictions = softmax(X_dec)
        self.logger.info("Transformer forward pass completed")
        return predictions
    
    def backprop(self, y_true, y_pred):
        self.logger.info("Computing transformer backpropagation")
        grad_output = 2 * (y_pred - y_true)
        return grad_output
    
    def update_parameters(self, grad_output):
        self.logger.info("Updating transformer parameters")
        self.encoder.backprop(grad_output)
        self.decoder.backprop(grad_output)
        self.linear_layer.backprop(grad_output)

    def train(self, X_enc, X_dec, y_true):
        self.logger.info(f"Starting training for {self.epochs} epochs")
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            y_pred = self.forward(X_enc, X_dec)
            loss = loss_function(y_true, y_pred)
            grad_output = self.backprop(y_true, y_pred)
            self.update_parameters(grad_output)
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}")
        return loss

if __name__ == "__main__":
    BATCH_SIZE = 128

    english_vocabulary_path = "/home/amaury-delille/Documents/machine_learning/ml-dl/attention-is-all-you-need/datasets/en_sents"
    english_vocabulary = Vocabulary(english_vocabulary_path)
    english_vocabulary_dict = english_vocabulary.run()

    vietnamese_vocabulary_path = "/home/amaury-delille/Documents/machine_learning/ml-dl/attention-is-all-you-need/datasets/vi_sents"
    vietnamese_vocabulary = Vocabulary(vietnamese_vocabulary_path)
    vietnamese_vocabulary_dict = vietnamese_vocabulary.run()

    english_input_embedding = InputEmbedding(x=english_vocabulary.sentences[:BATCH_SIZE], vocab_size=english_vocabulary.size, d_model=512, vocabulary=english_vocabulary)
    english_embeddings = english_input_embedding.forward()

    vietnamese_input_embedding = InputEmbedding(x=vietnamese_vocabulary.sentences[:BATCH_SIZE], vocab_size=vietnamese_vocabulary.size, d_model=512, vocabulary=vietnamese_vocabulary)
    vietnamese_embeddings = vietnamese_input_embedding.forward()
    
    print(english_embeddings.shape)
    print(vietnamese_embeddings.shape)
    
    transformer = Transformer(d_model=512, h=8, d_ff=2048, lr=0.01, epochs=10)
    transformer.train(english_embeddings, vietnamese_embeddings, vietnamese_vocabulary.sentences[:BATCH_SIZE])
