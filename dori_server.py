import json, os, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from dori_ai.response_engine import ResponseEngine

ROOT=Path(__file__).resolve().parent
CP=ROOT/'checkpoints'/'best.npz'; MP=Path(str(CP)+'.json')
if not CP.exists() or not MP.exists(): raise SystemExit('Missing checkpoints/best.npz or metadata.')
meta=json.loads(MP.read_text(encoding='utf-8'))
tok=BPETokenizer.load(ROOT/meta['tokenizer'])
model=load_model(str(CP),meta['model_config'])
bot=ResponseEngine(tok,model)

HTML='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dori AI</title><style>body{margin:0;background:#02051c;color:#eaf0ff;font-family:system-ui,sans-serif}main{max-width:900px;margin:auto;padding:28px}.card{background:#080d2e;border:1px solid #26336d;border-radius:20px;padding:20px;box-shadow:0 15px 60px #0008}.chat{min-height:65vh;display:flex;flex-direction:column;gap:12px}.m{padding:14px 16px;border-radius:15px;white-space:pre-wrap;line-height:1.5;max-width:85%}.u{background:#16265c;align-self:flex-end}.d{background:#0d1439;align-self:flex-start}form{display:flex;gap:10px;margin-top:16px}input{flex:1;padding:15px;border-radius:12px;border:1px solid #34447f;background:#03071f;color:white;font-size:16px}button{padding:15px 20px;border:0;border-radius:12px;cursor:pointer}</style></head><body><main><div class="card"><h1>🐶 Dori AI</h1><div id="c" class="chat"></div><form><input id="i" placeholder="돌이에게 말해보세요…" autocomplete="off"><button>전송</button></form></div></main><script>const c=document.querySelector('#c'),f=document.querySelector('form'),i=document.querySelector('#i');function add(t,k){const x=document.createElement('div');x.className='m '+k;x.textContent=t;c.append(x);scrollTo(0,document.body.scrollHeight)}f.onsubmit=async e=>{e.preventDefault();const t=i.value.trim();if(!t)return;i.value='';add(t,'u');try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});const d=await r.json();add(d.answer||d.error,'d')}catch(e){add('서버에 연결할 수 없어.','d')}};</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def send_json(self,status,obj):
        data=json.dumps(obj,ensure_ascii=False).encode('utf-8'); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path=='/health': self.send_json(200,{'status':'ok','service':'dori-ai','version':meta.get('version'),'model':meta.get('model_config')}); return
        data=HTML.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_POST(self):
        if self.path!='/chat': self.send_json(404,{'error':'not found'}); return
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n>16384: raise ValueError('request too large')
            d=json.loads(self.rfile.read(n)); text=str(d.get('text','')).strip()
            if not text or len(text)>1000: raise ValueError('text must be 1-1000 characters')
            answer=bot.reply(text)
            self.send_json(200,{'answer':answer})
        except Exception as e: self.send_json(400,{'error':str(e)})
    def log_message(self,fmt,*args): return

if __name__=='__main__':
    host=os.getenv('DORI_HOST','127.0.0.1'); port=int(os.getenv('PORT',os.getenv('DORI_PORT','8000')))
    print(f'Dori AI server listening on {host}:{port}')
    ThreadingHTTPServer((host,port),Handler).serve_forever()
