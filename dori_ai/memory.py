import json, math, re, time
from pathlib import Path
class MemoryStore:
    def __init__(self,path='memory/dori_memory.json'):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.items=[]; self.load()
    def load(self):
        if self.path.exists():
            try:self.items=json.loads(self.path.read_text(encoding='utf-8'))
            except Exception:self.items=[]
    def save(self): self.path.write_text(json.dumps(self.items,ensure_ascii=False,indent=2),encoding='utf-8')
    def _terms(self,s): return set(re.findall(r'[\w가-힣]+',s.lower()))
    def add(self,text,kind='fact'):
        text=text.strip()
        if not text:return
        self.items.append({'text':text,'kind':kind,'time':time.time()}); self.save()
    def search(self,query,k=5):
        qt=self._terms(query); scored=[]
        for x in self.items:
            st=self._terms(x['text']); overlap=len(qt&st)
            if query.strip() and query.strip() in x['text']: overlap += 2
            score=overlap/(math.sqrt(len(qt)*len(st))+1e-9)
            if score: scored.append((score,x))
        scored.sort(key=lambda z:z[0],reverse=True); return [x for _,x in scored[:k]]
    def context(self,query,k=5):
        return '\n'.join('- '+x['text'] for x in self.search(query,k))
