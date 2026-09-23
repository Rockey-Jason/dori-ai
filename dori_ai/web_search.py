"""Optional public-web retrieval without an AI API or pretrained model.

Uses DuckDuckGo's HTML endpoint and only returns short result snippets/links.
Network failures are treated as a normal no-result condition.
"""
import html,re,urllib.parse,urllib.request
from html.parser import HTMLParser

class _Parser(HTMLParser):
    def __init__(self):
        super().__init__(); self.items=[]; self._a=None; self._txt=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=="a" and ("result__a" in d.get("class","") or "result-link" in d.get("class","")):
            self._a=d.get("href"); self._txt=[]
    def handle_data(self,data):
        if self._a:self._txt.append(data)
    def handle_endtag(self,tag):
        if tag=="a" and self._a:
            title=" ".join("".join(self._txt).split())
            if title:self.items.append((title,self._a))
            self._a=None; self._txt=[]

def search(query,limit=5,timeout=4):
    q=str(query).strip()
    if not q:return []
    url="https://html.duckduckgo.com/html/?q="+urllib.parse.quote_plus(q)
    req=urllib.request.Request(url,headers={"User-Agent":"DoriAI/2.3 (+https://dori-ai-u3kf.onrender.com)"})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            raw=r.read().decode("utf-8","ignore")
    except Exception:
        return []
    p=_Parser(); p.feed(raw)
    out=[]; seen=set()
    for title,href in p.items:
        href=html.unescape(href)
        if href.startswith("//"):href="https:"+href
        if href in seen:continue
        seen.add(href)
        out.append({"title":title,"url":href})
        if len(out)>=limit:break
    return out

def format_results(query,results):
    if not results:return None
    lines=[f"웹 검색 결과야. 질문: {query}",""]
    for i,r in enumerate(results,1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['url']}")
    lines.append("")
    lines.append("검색 결과의 제목과 원문 링크를 확인하고, 중요한 사실은 원문에서 다시 검증해줘.")
    return "\n".join(lines)
