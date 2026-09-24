import numpy as np

class Adam:
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr, self.beta1, self.beta2, self.eps, self.weight_decay = lr, beta1, beta2, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0
    def step(self):
        self.t += 1
        for i,p in enumerate(self.parameters):
            g = p.grad
            if self.weight_decay: g = g + self.weight_decay * p.data
            norm = np.linalg.norm(g)
            if norm > 1.0: g = g / norm
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*g
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*(g*g)
            mh = self.m[i] / (1-self.beta1**self.t)
            vh = self.v[i] / (1-self.beta2**self.t)
            p.data -= self.lr * mh / (np.sqrt(vh)+self.eps)
