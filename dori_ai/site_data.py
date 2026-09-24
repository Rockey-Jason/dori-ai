"""Read-only, live bridge to Dori site's Supabase data."""
import json, os, urllib.parse, urllib.request

class SiteData:
    def __init__(self, access_token=None):
        self.url=os.getenv("SUPABASE_URL","").rstrip("/")
        self.key=os.getenv("SUPABASE_ANON_KEY","")
        self.access_token=access_token

    def with_token(self, token): return SiteData(token or self.access_token)

    def _get(self, table, params, timeout=6):
        if not self.url or not self.key: return None
        q=urllib.parse.urlencode(params, doseq=True)
        bearer=self.access_token or self.key
        req=urllib.request.Request(f"{self.url}/rest/v1/{table}?{q}", headers={"apikey":self.key,"Authorization":f"Bearer {bearer}","Accept":"application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())
        except Exception: return None

    def is_admin(self,user_id=None):
        if not user_id:return False
        rows=self._get("users",{"select":"is_admin,user_level","user_id":f"eq.{user_id}","limit":"1"}) or []
        return bool(rows and rows[0].get("is_admin")) and int(rows[0].get("user_level") or 0)>=10

    def readable_news_limit(self,user_id=None):
        if not user_id:return 0
        rows=self._get("users",{"select":"read_dori_news","user_id":f"eq.{user_id}","limit":"1"}) or []
        try:return max(0,int(rows[0].get("read_dori_news") or 0)) if rows else 0
        except Exception:return 0

    def public_news_all(self):
        return self._get("rockey_news",{"select":"news_number,rockey_news,question,question_type,choice1,choice2,choice3,choice4,choice5","order":"news_number.asc"},timeout=10) or []

    def public_news(self,number):
        try:n=int(number)
        except Exception:return None
        rows=self._get("rockey_news",{"select":"news_number,rockey_news,question,question_type,choice1,choice2,choice3,choice4,choice5","news_number":f"eq.{n}","limit":"1"}) or []
        return rows[0] if rows else None

    def news(self,number,user_id=None):
        try:n=int(number)
        except Exception:return None
        limit=self.readable_news_limit(user_id)
        if n<1 or n>limit:return {"denied":True,"number":n,"max_readable":limit}
        rows=self._get("rockey_news",{"select":"news_number,rockey_news,question,question_type,choice1,choice2,choice3,choice4,choice5,answer","news_number":f"eq.{n}","limit":"1"}) or []
        return rows[0] if rows else None

    def stock(self,query=None):
        rows=self._get("dori_stocks",{"select":"id,ticker,name,description,risk_label,current_price,previous_price,change_amount,change_percent,is_active,characteristics,available_shares","is_active":"eq.true","order":"id.asc"},timeout=8) or []
        if not query:return rows
        q=str(query).lower(); return [x for x in rows if q in str(x.get("name","")).lower() or q in str(x.get("ticker","")).lower()]

    def profile(self,user_id=None):
        if not user_id:return None
        rows=self._get("users",{"select":"name,user_level,exp_level,doldolcoin,exp,equipped_title,read_dori_news","user_id":f"eq.{user_id}","limit":"1"}) or []
        return rows[0] if rows else None

    def site_summary(self):
        news=self.public_news_all(); stocks=self.stock()
        return {"news_count":len(news),"latest_news":max([int(x.get("news_number") or 0) for x in news],default=0),"stock_count":len(stocks),"stocks":stocks}
