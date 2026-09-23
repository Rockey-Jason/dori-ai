import json
from pathlib import Path
from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from dori_ai.response_engine import ResponseEngine

def main():
 cp=Path('checkpoints/best.npz'); mp=Path(str(cp)+'.json')
 if not cp.exists() or not mp.exists(): raise SystemExit('학습된 모델이 없습니다. build_and_train_final.bat을 먼저 실행하세요.')
 meta=json.loads(mp.read_text(encoding='utf-8')); tok=BPETokenizer.load(meta['tokenizer']); model=load_model(str(cp),meta['model_config']); bot=ResponseEngine(tok,model)
 print('🐶 Dori AI FINAL online | /exit 종료 | /remember 내용 저장 | /memory 검색')
 while True:
  try:u=input('\nYou: ').strip()
  except (EOFError,KeyboardInterrupt): print('\nDori: 다음에 또 만나자! 🐶'); break
  if not u: continue
  if u=='/exit': print('Dori: 다음에 또 만나자! 🐶'); break
  if u.startswith('/remember '): bot.memory.add(u[10:]); print('Dori: 기억했어! 🧠'); continue
  if u=='/memory' or u.startswith('/memory '):
   q=u[8:].strip(); print(bot.memory.context(q) if q else ('\n'.join('- '+x['text'] for x in bot.memory.items[-10:]) or '(기억 없음)')); continue
  print('Dori:',bot.reply(u))
if __name__=='__main__': main()
