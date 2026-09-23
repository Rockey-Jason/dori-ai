import numpy as np


def _unbroadcast(grad, shape):
    g = grad
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g


def concat(tensors, axis=-1):
    tensors = list(tensors)
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, any(t.requires_grad for t in tensors), tuple(tensors), 'concat')
    widths = [t.data.shape[axis] for t in tensors]

    def backward():
        start = 0
        sl = [slice(None)] * out.grad.ndim
        for t, width in zip(tensors, widths):
            if t.requires_grad:
                sl[axis] = slice(start, start + width)
                t.grad += out.grad[tuple(sl)]
            start += width
    out._backward = backward
    return out


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def zero_grad(self):
        self.grad.fill(0)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), '+')
        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = backward
        return out
    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), '*')
        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = backward
        return out
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def pow(self, power):
        out = Tensor(self.data ** power, self.requires_grad, (self,), 'pow')
        def backward():
            if self.requires_grad:
                self.grad += power * np.power(self.data, power - 1) * out.grad
        out._backward = backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other), '@')
        def backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), 'sum')
        def backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted([a if a >= 0 else self.data.ndim + a for a in axes]):
                    g = np.expand_dims(g, ax)
            self.grad += np.ones_like(self.data) * g
        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        elif isinstance(axis, tuple):
            n = int(np.prod([self.data.shape[a] for a in axis]))
        else:
            n = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def exp(self):
        value = np.exp(np.clip(self.data, -60, 60))
        out = Tensor(value, self.requires_grad, (self,), 'exp')
        def backward():
            if self.requires_grad:
                self.grad += value * out.grad
        out._backward = backward
        return out

    def log(self):
        safe = np.maximum(self.data, 1e-12)
        out = Tensor(np.log(safe), self.requires_grad, (self,), 'log')
        def backward():
            if self.requires_grad:
                self.grad += out.grad / safe
        out._backward = backward
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(value, self.requires_grad, (self,), 'tanh')
        def backward():
            if self.requires_grad:
                self.grad += (1 - value * value) * out.grad
        out._backward = backward
        return out

    def relu(self):
        value = np.maximum(self.data, 0)
        out = Tensor(value, self.requires_grad, (self,), 'relu')
        def backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = backward
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, self.requires_grad, (self,), 'transpose')
        def backward():
            if self.requires_grad:
                self.grad += out.grad.T
        out._backward = backward
        return out

    def backward(self):
        self.grad = np.ones_like(self.data)
        topo, visited = [], set()
        def build(v):
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
        build(self)
        for node in reversed(topo):
            node._backward()
