import numpy as np
from .tensor import Tensor
from .layers import Parameter, Linear, LayerNorm, FeedForward
from .attention import MultiHeadSelfAttention

class TokenEmbedding:
    def __init__(self, vocab_size, dim):
        self.weight = Parameter(np.random.randn(vocab_size, dim) * 0.02)
    def __call__(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        out = Tensor(self.weight.data[ids], True, (self.weight,), 'embedding')
        def backward():
            np.add.at(self.weight.grad, ids, out.grad)
        out._backward = backward
        return out
    def parameters(self): return [self.weight]

class PositionEmbedding:
    def __init__(self, max_len, dim):
        self.weight = Parameter(np.random.randn(max_len, dim) * 0.02)
    def __call__(self, length):
        out = Tensor(self.weight.data[:length], True, (self.weight,), 'position')
        def backward():
            self.weight.grad[:length] += out.grad
        out._backward = backward
        return out
    def parameters(self): return [self.weight]

class TransformerBlock:
    def __init__(self, dim, heads, ff_dim):
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim)
    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
    def parameters(self):
        return self.norm1.parameters() + self.attn.parameters() + self.norm2.parameters() + self.ff.parameters()

class DoriTransformer:
    def __init__(self, vocab_size, max_len=64, dim=64, heads=4, ff_dim=256, layers=4):
        self.vocab_size, self.max_len = vocab_size, max_len
        self.dim, self.heads, self.ff_dim, self.layers_count = dim, heads, ff_dim, layers
        self.token_embedding = TokenEmbedding(vocab_size, dim)
        self.position_embedding = PositionEmbedding(max_len, dim)
        self.blocks = [TransformerBlock(dim, heads, ff_dim) for _ in range(layers)]
        self.norm = LayerNorm(dim)
        self.lm_head = Linear(dim, vocab_size)
    def __call__(self, ids):
        if len(ids) > self.max_len: raise ValueError('sequence is longer than max_len')
        x = self.token_embedding(ids) + self.position_embedding(len(ids))
        for block in self.blocks: x = block(x)
        return self.lm_head(self.norm(x))
    def parameters(self):
        p = self.token_embedding.parameters() + self.position_embedding.parameters()
        for b in self.blocks: p += b.parameters()
        return p + self.norm.parameters() + self.lm_head.parameters()
    def zero_grad(self):
        for p in self.parameters(): p.zero_grad()
    def config(self):
        return dict(vocab_size=self.vocab_size, max_len=self.max_len, dim=self.dim, heads=self.heads, ff_dim=self.ff_dim, layers=self.layers_count)
    def save(self, path):
        np.savez(path, **{f'p{i}': p.data for i,p in enumerate(self.parameters())})
    def load_weights(self, path):
        data = np.load(path)
        params = self.parameters()
        if len(data.files) != len(params): raise ValueError('checkpoint parameter count mismatch')
        for i,p in enumerate(params): p.data[...] = data[f'p{i}']

def load_model(path, config):
    model = DoriTransformer(**config)
    model.load_weights(path)
    return model
