import argparse,json,math
from pathlib import Path
import numpy as np
from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model

def safe_softmax(logits,temp):
 x=np.asarray(logits,dtype=np.float64).reshape(-1); x=np.nan_to_num(x,nan=0.0,posinf=50.0,neginf=-50.0)
 t=max(float(temp),0.05); x=x/t; x-=np.max(x); p=np.exp(np.clip(x,-50,0)); s=float(p.sum())
 if not np.isfinite(s) or s<=0:return np.full(len(x),1/len(x))
 p/=s; p/=p.sum(); return p

def sample(logits,temp=.7,top_k=20,top_p=.9,rng=None):
 rng=rng or np.random.default_rng(); p=safe_softmax(logits,temp)
 k=min(max(int(top_k),0),len(p))
 if k and k<len(p):
  ids=np.argpartition(p,-k)[-k:]; q=np.zeros_like(p); q[ids]=p[ids]; p=q/q.sum()
 nucleus=min(max(float(top_p),0.0),1.0)
 if nucleus<1:
  ids=np.argsort(p)[::-1]; c=np.cumsum(p[ids]); n=max(1,int(np.searchsorted(c,nucleus,'left'))+1); q=np.zeros_like(p); q[ids[:n]]=p[ids[:n]]; p=q/q.sum()
 p=np.maximum(p,0); p/=p.sum(); p[np.argmax(p)]+=1-p.sum(); return int(rng.choice(len(p),p=p))

def load_bundle(cp='checkpoints/best.npz'):
 cp=Path(cp); mp=Path(str(cp)+'.json')
 if not cp.exists() or not mp.exists(): raise FileNotFoundError(f'학습된 checkpoint가 없습니다: {cp}')
 meta=json.loads(mp.read_text(encoding='utf-8')); tok=BPETokenizer.load(meta['tokenizer']); model=load_model(str(cp),meta['model_config']); return tok,model

def generate(prompt,tok,model,tokens=100,temp=.7,top_k=20,top_p=.9,seed=None):
 ids=tok.encode(prompt); out=list(ids); rng=np.random.default_rng(seed)
 for _ in range(int(tokens)):
  x=np.asarray(out[-model.max_len:],dtype=np.int64); logits=model(x).data[-1]
  nxt=sample(logits,temp,top_k,top_p,rng); out.append(nxt)
  tail=tok.decode(out[len(ids):])
  if '<eos>' in tail or '<user>' in tail: break
 return tok.decode(out[len(ids):]).split('<eos>')[0].split('<user>')[0].strip()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='checkpoints/best.npz'); ap.add_argument('--prompt',default='돌이는 뭐야?'); ap.add_argument('--tokens',type=int,default=100); ap.add_argument('--temperature',type=float,default=.7); ap.add_argument('--top-k',type=int,default=20); ap.add_argument('--top-p',type=float,default=.9); a=ap.parse_args()
 tok,model=load_bundle(a.checkpoint); print(generate(a.prompt,tok,model,a.tokens,a.temperature,a.top_k,a.top_p))
if __name__=='__main__': main()
