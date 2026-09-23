import json, os, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from dori_ai.response_engine import ResponseEngine
ROOT=Path(__file__).resolve().parent; CP=ROOT/'checkpoints'/'best.npz'; MP=Path(str(CP)+'.json')
if not CP.exists() or not MP.exists(): raise SystemExit('Missing checkpoints/best.npz or metadata.')
meta=json.loads(MP.read_text(encoding='utf-8')); tok=BPETokenizer.load(ROOT/meta['tokenizer']); model=load_model(str(CP),meta['model_config']); bot=ResponseEngine(tok,model)
CORS={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Methods':'GET, POST, OPTIONS','Access-Control-Allow-Headers':'Content-Type, Accept, Authorization','Access-Control-Max-Age':'86400'}
HTML='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dori AI</title><style>body{margin:0;background:#02051c;color:#eaf0ff;font-family:system-ui,sans-serif}main{max-width:900px;margin:auto;padding:28px}.card{background:#080d2e;border:1px solid #26336d;border-radius:20px;padding:20px;box-shadow:0 15px 60px #0008}.chat{min-height:65vh;display:flex;flex-direction:column;gap:12px}.m{padding:14px 16px;border-radius:15px;white-space:pre-wrap;line-height:1.5;max-width:85%}.u{background:#16265c;align-self:flex-end}.d{background:#0d1439;align-self:flex-start}form{display:flex;gap:10px;margin-top:16px}input{flex:1;padding:15px;border-radius:12px;border:1px solid #34447f;background:#03071f;color:white;font-size:16px}button{padding:15px 20px;border:0;border-radius:12px;cursor:pointer}</style><main><div class="card"><h1>🐶 Dori AI</h1><div id="c" class="chat"></div><form><input id="i" placeholder="돌이에게 말해보세요…" autocomplete="off"><button>전송</button></form></div></main><script>const c=document.querySelector('#c'),f=document.querySelector('form'),i=document.querySelector('#i');function add(t,k){const x=document.createElement('div');x.className='m '+k;x.textContent=t;c.append(x);scrollTo(0,document.body.scrollHeight)}f.onsubmit=async e=>{e.preventDefault();const t=i.value.trim();if(!t)return;i.value='';add(t,'u');try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});const d=await r.json();add(d.answer||d.error,'d')}catch(e){add('서버 오류가 발생했어.','d')}};</script></html>'''
class H(BaseHTTPRequestHandler):
 def cors(self):
  for k,v in CORS.items(): self.send_header(k,v)
 def send(self,c,obj,typ='application/json; charset=utf-8'):
  b=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(c); self.send_header('Content-Type',typ); self.send_header('Content-Length',str(len(b))); self.send_header('Cache-Control','no-store'); self.cors(); self.end_headers(); self.wfile.write(b)
 def do_OPTIONS(self): self.send_response(204); self.cors(); self.end_headers()
 def do_GET(self):
  if self.path.split('?',1)[0]=='/health': self.send(200,{'status':'ok','service':'dori-ai','version':meta.get('version'),'knowledge_entries':bot.kb.size(),'web_search':bot.web_enabled,'model':meta.get('model_config')}); return
  self.send(200,HTML,'text/html; charset=utf-8')
 def do_POST(self):
  if self.path.split('?',1)[0]!='/chat': self.send(404,{'error':'not found'}); return
  try:
   n=int(self.headers.get('Content-Length','0'))
   if n>16384: raise ValueError('request too large')
   d=json.loads(self.rfile.read(n)); text=str(d.get('message',d.get('text',''))).strip(); uid=str(d.get('user_id','')).strip() or None
   if not text or len(text)>2000: raise ValueError('text must be 1-2000 characters')
   a=bot.reply(text,user_id=uid); self.send(200,{'answer':a,'mode':str(d.get('mode','fast'))})
  except Exception as e: self.send(400,{'error':str(e)})
 def log_message(self,*a): pass
if __name__=='__main__':
 host=os.getenv('DORI_HOST','0.0.0.0'); port=int(os.getenv('PORT',os.getenv('DORI_PORT','8000'))); print(f'Dori AI server listening on {host}:{port}',flush=True); ThreadingHTTPServer((host,port),H).serve_forever()
