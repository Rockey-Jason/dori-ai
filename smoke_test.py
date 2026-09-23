from pathlib import Path
import json, numpy as np
from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from dori_ai.response_engine import ResponseEngine
cp=Path('checkpoints/best.npz'); meta=json.loads(Path(str(cp)+'.json').read_text(encoding='utf-8'))
tok=BPETokenizer.load(meta['tokenizer']); model=load_model(str(cp),meta['model_config']); bot=ResponseEngine(tok,model)
assert tok.decode(tok.encode('돌이는 웰시코기 인형 캐릭터야. 🐶'))=='돌이는 웰시코기 인형 캐릭터야. 🐶'
assert '웰시코기' in bot.reply('돌이는 뭐야?')
assert '12×12' in bot.reply('돌이전쟁이 뭐야?')
assert bot.reply('안녕').startswith('안녕!')
print('SMOKE TEST: PASS')
