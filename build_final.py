from pathlib import Path
from dori_ai.corpus_engine import build_datasets
from dori_ai.bpe_tokenizer import BPETokenizer
root=Path('.')
text_len=build_datasets('data'); text=Path('data/corpus/training.txt').read_text(encoding='utf-8')
tok=BPETokenizer(512).train(text); tok.save('data/tokenizer.json')
# Hard safety checks: tokenizer must round-trip representative Unicode and tags.
checks=['돌이는 웰시코기 인형 캐릭터야.','안녕 🐶','<system>너는 Dori AI다.<user>돌이는 뭐야?<dori>정답<eos>']
for s in checks:
 if tok.decode(tok.encode(s))!=s: raise RuntimeError('Tokenizer round-trip failed')
print(f'Corpus: {text_len:,} chars | tokens: {len(tok.encode(text)):,} | vocab: {tok.vocab_size} | merges: {len(tok.merges)}')
