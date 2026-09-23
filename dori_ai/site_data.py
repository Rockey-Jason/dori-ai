"""Read-only bridge from Dori AI to the Dori site's Supabase data.
Only explicitly requested public Dori data is exposed. News access is capped
by the requesting user's read_dori_news value.
"""
import json,os,urllib.parse,urllib.request

class SiteData:
    def __init__(self):
        self.url=os.getenv("SUPABASE_URL","").rstrip("/")
        self.key=os.getenv("SUPABASE_ANON_KEY","")
    def _get(self,table,params):
        if not self.url or not self.key:return None
        q=urllib.parse.urlencode(params,doseq=True)
        req=urllib.request.Request(f"{self.url}/rest/v1/{table}?{q}",headers={"apikey":self.key,"Authorization":f"Bearer {self.key}","Accept":"application/json"})
        try:
            with urllib.request.urlopen(req,timeout=5) as r:return json.loads(r.read().decode())
        except Exception:return None
    def readable_news_limit(self,user_id=None):
        if not user_id:return 0
        rows=self._get("users",{"select":"read_dori_news","user_id":f"eq.{user_id}","limit":"1"})
        if not rows:return 0
        try:return max(0,int(rows[0].get("read_dori_news") or 0))
        except Exception:return 0
    def news(self,number,user_id=None):
        try:n=int(number)
        except Exception:return None
        limit=self.readable_news_limit(user_id)
        if n<1 or n>limit:return {"denied":True,"number":n,"max_readable":limit}
        rows=self._get("rockey_news",{"select":"news_number,rockey_news,question,question_type,choice1,choice2,choice3,choice4,choice5,answer","news_number":f"eq.{n}","limit":"1"})
        return rows[0] if rows else None
    def stock(self,query=None):
        params={"select":"ticker,name,description,risk_label,current_price,previous_price,change_amount,change_percent,is_active,characteristics,available_shares","is_active":"eq.true","order":"id.asc"}
        rows=self._get("dori_stocks",params) or []
        if not query:return rows
        q=str(query).lower()
        return [x for x in rows if q in str(x.get("name","")).lower() or q in str(x.get("ticker","")).lower()]
