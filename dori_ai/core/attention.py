import numpy as np
from .tensor import Tensor, concat
from .layers import Linear

class MultiHeadSelfAttention:
    def __init__(self, dim, num_heads):
        if dim % num_heads != 0:
            raise ValueError('embedding_dim must be divisible by num_heads')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = [Linear(dim, self.head_dim) for _ in range(num_heads)]
        self.k = [Linear(dim, self.head_dim) for _ in range(num_heads)]
        self.v = [Linear(dim, self.head_dim) for _ in range(num_heads)]
        self.out = Linear(dim, dim)

    def _masked_softmax(self, scores):
        t = scores.data.shape[0]
        mask = np.triu(np.ones((t, t)), k=1)
        masked = Tensor(scores.data + np.where(mask == 1, -1e9, 0.0), scores.requires_grad, (scores,), 'causal_mask')
        def backward():
            if scores.requires_grad:
                scores.grad += masked.grad
        masked._backward = backward
        shifted = masked + (-np.max(masked.data, axis=1, keepdims=True))
        ex = shifted.exp()
        return ex / ex.sum(axis=1, keepdims=True)

    def __call__(self, x):
        heads = []
        for q_layer, k_layer, v_layer in zip(self.q, self.k, self.v):
            q, k, v = q_layer(x), k_layer(x), v_layer(x)
            scores = (q @ k.T) / np.sqrt(self.head_dim)
            weights = self._masked_softmax(scores)
            heads.append(weights @ v)
        return self.out(concat(heads, axis=1))

    def parameters(self):
        params = []
        for layer in self.q + self.k + self.v:
            params += layer.parameters()
        params += self.out.parameters()
        return params
