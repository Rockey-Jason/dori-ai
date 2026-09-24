import json,hashlib,sys
from pathlib import Path
import numpy as np
from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from dori_ai.retrieval import LocalKnowledge
from generate_final import sample

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 t=BPETokenizer.load('data/tokenizer.json')
 for s in ['돌이는 웰시코기 인형 캐릭터야.','안녕 🐶','<system>hello<user>돌이<dori>좋아<eos>']:
  assert t.decode(t.encode(s))==s
 cp=Path('checkpoints/best.npz'); mp=Path(str(cp)+'.json'); assert cp.exists() and mp.exists()
 meta=json.loads(mp.read_text(encoding='utf-8')); assert sha('data/tokenizer.json')==meta['tokenizer_sha256']
 m=load_model(str(cp),meta['model_config']); ids=t.encode('돌이는 뭐야?'); logits=m(np.asarray(ids,dtype=np.int64)).data[-1]; assert np.all(np.isfinite(logits)); assert 0<=sample(logits,.7,16,.9)<t.vocab_size
 kb=LocalKnowledge(); assert kb.answer('돌이는 뭐야?')
 print('Tokenizer: PASS'); print('Checkpoint integrity: PASS'); print('Inference finite: PASS'); print('Knowledge retrieval: PASS'); print('ALL FINAL TESTS PASSED')
if __name__=='__main__': main()
