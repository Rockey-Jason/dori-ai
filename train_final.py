import argparse,json,os,hashlib,time
from pathlib import Path
import numpy as np
from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import DoriTransformer
from dori_ai.core.optimizer import Adam
from dori_ai.trainer import Trainer

def sha256(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--epochs',type=int,default=500); ap.add_argument('--seq-len',type=int,default=96)
 ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--dim',type=int,default=64)
 ap.add_argument('--heads',type=int,default=4); ap.add_argument('--layers',type=int,default=3)
 ap.add_argument('--ff-dim',type=int,default=256); ap.add_argument('--lr',type=float,default=3e-4)
 ap.add_argument('--seed',type=int,default=20260916); ap.add_argument('--data',default='data/corpus/v23_training.txt')
 ap.add_argument('--vocab-size',type=int,default=768)
 ap.add_argument('--retrain-tokenizer',action='store_true')
 a=ap.parse_args()
 if a.dim%a.heads: raise SystemExit('dim must be divisible by heads')
 np.random.seed(a.seed); os.makedirs('checkpoints',exist_ok=True)
 tok_path=Path('data/tokenizer.json'); data_path=Path(a.data)
 if a.retrain_tokenizer or not tok_path.exists():
  text=data_path.read_text(encoding='utf-8'); tok=BPETokenizer(a.vocab_size).train(text); tok.save(tok_path)
 else: tok=BPETokenizer.load(tok_path); text=data_path.read_text(encoding='utf-8')
 tokens=tok.encode(text)
 model=DoriTransformer(tok.vocab_size,a.seq_len,a.dim,a.heads,a.ff_dim,a.layers)
 opt=Adam(model.parameters(),lr=a.lr,weight_decay=1e-5)
 tr=Trainer(model,opt,tokens,a.seq_len,a.batch_size,val_fraction=.12,seed=a.seed)
 best=float('inf'); start=time.time()
 print(f'Corpus: {len(text):,} chars | tokens: {len(tokens):,} | vocab: {tok.vocab_size}')
 print(f'Model: dim={a.dim}, heads={a.heads}, layers={a.layers}, ff={a.ff_dim}, context={a.seq_len}')
 for e in range(1,a.epochs+1):
  tl=tr.train_step()
  if e==1 or e%25==0 or e==a.epochs:
   vl=tr.evaluate(batches=8)
   meta={'format_version':3,'model_config':model.config(),'tokenizer':'data/tokenizer.json','tokenizer_sha256':sha256(tok_path),'epoch':e,'train_loss':tl,'val_loss':vl,'version':'2.3.0'}
   model.save('checkpoints/latest.npz'); Path('checkpoints/latest.npz.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
   if np.isfinite(vl) and vl<best:
    best=vl; model.save('checkpoints/best.npz'); Path('checkpoints/best.npz.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
   print(f'Epoch {e:4d} | train {tl:.4f} | val {vl:.4f} | best {best:.4f}')
 print(f'Training complete in {time.time()-start:.1f}s')

if __name__=='__main__': main()
