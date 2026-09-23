"""Small, deterministic mini-batch trainer for the hand-written Transformer."""
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, tokens, seq_len=96, batch_size=8, val_fraction=.12, seed=0):
        self.model=model; self.optimizer=optimizer
        self.tokens=np.asarray(tokens,dtype=np.int64)
        self.seq_len=max(8,int(seq_len)); self.batch_size=max(1,int(batch_size))
        cut=max(self.seq_len+2,int(len(self.tokens)*(1-float(val_fraction))))
        cut=min(cut,len(self.tokens)-1)
        self.train=self.tokens[:cut]; self.val=self.tokens[cut:]
        self.rng=np.random.default_rng(seed)
    def _batch(self, arr):
        if len(arr)<=self.seq_len+1: raise ValueError('corpus is too small')
        xs=[]; ys=[]
        for _ in range(self.batch_size):
            s=int(self.rng.integers(0,len(arr)-self.seq_len))
            xs.append(arr[s:s+self.seq_len]); ys.append(arr[s+1:s+self.seq_len+1])
        return xs,ys
    def _loss_and_backward(self, arr):
        self.model.zero_grad(); losses=[]
        xs,ys=self._batch(arr)
        for ids,target in zip(xs,ys):
            logits=self.model(ids)
            z=logits.data-np.max(logits.data,axis=1,keepdims=True)
            p=np.exp(z); p/=np.sum(p,axis=1,keepdims=True)
            idx=np.arange(len(target)); loss=-np.log(np.maximum(p[idx,target],1e-12)).mean()
            losses.append(float(loss))
            g=p; g[idx,target]-=1.0; g/=len(target)
            logits.backward(g/len(xs))
        self.optimizer.step()
        return float(np.mean(losses))
    def train_step(self): return self._loss_and_backward(self.train)
    def evaluate(self,batches=8):
        losses=[]
        for _ in range(max(1,int(batches))):
            xs,ys=self._batch(self.val)
            for ids,target in zip(xs,ys):
                logits=self.model(ids)
                z=logits.data-np.max(logits.data,axis=1,keepdims=True)
                p=np.exp(z); p/=np.sum(p,axis=1,keepdims=True)
                idx=np.arange(len(target)); losses.append(float(-np.log(np.maximum(p[idx,target],1e-12)).mean()))
        return float(np.mean(losses))
