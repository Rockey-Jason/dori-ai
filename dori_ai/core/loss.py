import numpy as np
from .tensor import Tensor

def cross_entropy(logits, targets):
    targets = np.asarray(targets, dtype=np.int64)
    # Stable log-softmax: max is treated as a constant shift.
    shifted = logits + (-np.max(logits.data, axis=1, keepdims=True))
    logsumexp = shifted.exp().sum(axis=1, keepdims=True).log()
    log_probs = shifted - logsumexp
    gathered = Tensor(log_probs.data[np.arange(len(targets)), targets], log_probs.requires_grad, (log_probs,), 'gather')
    def backward():
        if log_probs.requires_grad:
            np.add.at(log_probs.grad, (np.arange(len(targets)), targets), gathered.grad)
    gathered._backward = backward
    return (-gathered).mean()
