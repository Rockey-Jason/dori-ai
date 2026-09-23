import numpy as np
from .tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)

class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.weight = Parameter(np.random.randn(in_features, out_features) * scale)
        self.bias = Parameter(np.zeros(out_features))
    def __call__(self, x):
        return x @ self.weight + self.bias
    def parameters(self):
        return [self.weight, self.bias]

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Parameter(np.ones(dim))
        self.beta = Parameter(np.zeros(dim))
    def __call__(self, x):
        mean = x.mean(axis=1, keepdims=True)
        centered = x - mean
        variance = (centered * centered).mean(axis=1, keepdims=True)
        inv_std = (variance + self.eps).pow(-0.5)
        return centered * inv_std * self.gamma + self.beta
    def parameters(self):
        return [self.gamma, self.beta]

class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.fc1 = Linear(dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, dim)
    def __call__(self, x):
        x = self.fc1(x)
        # GELU tanh approximation
        inner = x + 0.044715 * x * x * x
        x = 0.5 * x * (1.0 + (np.sqrt(2.0 / np.pi) * inner).tanh())
        return self.fc2(x)
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()
