import re, os
from .retrieval import LocalKnowledge
from .memory import MemoryStore
from .dialogue import DialogueManager
from .web_search import search as web_search, format_results
from .site_data import SiteData
from .language import detect, normalize_query
from generate_final import generate

class ResponseEngine:
    def __init__(self, tok, model):
        self.tok, self.model = tok, model
        self.dialogues = {}
        self.kb = LocalKnowledge()
        self.site = SiteData()
        self.web_enabled = os.getenv("DORI_WEB_SEARCH", "1").lower() not in ("0","false","off")
        self.smalltalk = {
            "안녕":"안녕! 나는 돌이야 🐶 오늘은 무슨 이야기를 해볼까?",
            "안녕하세요":"안녕! 나는 돌이야 🐶 무엇을 도와줄까?",
            "고마워":"천만에! 도움이 되었다면 기뻐! 🐶",
            "hello":"Hello! I'm Dori AI. 🐶","hi":"Hi! I'm Dori AI. 🐶",
            "hola":"¡Hola! Soy Dori AI. 🐶","bonjour":"Bonjour ! Je suis Dori AI. 🐶",
            "こんにちは":"こんにちは！Dori AIだよ。🐶","你好":"你好！我是Dori AI。🐶"
        }

    def _dialogue(self, user_id):
        key = str(user_id or "anonymous")
        if key not in self.dialogues:
            self.dialogues[key] = DialogueManager(MemoryStore(f"memory/users/{key}.json"))
        return self.dialogues[key]

    @staticmethod
    def _bad(text):
        if not text or len(text.strip()) < 2: return True
        t=text.strip()
        if any(x in t.lower() for x in ("<system>","<user>","<dori>","<eos>")): return True
        if re.search(r"(.)\1{7,}", t): return True
        words=re.findall(r"[가-힣A-Za-z0-9]+",t)
        if len(words)>=10 and len(set(words))/len(words)<.55: return True
        return False

    @staticmethod
    def _news_number(u):
        patterns=[
            r"(?:돌이\s*신문|신문|newspaper|news|新聞|报纸)\s*(?:제\s*)?(\d+)\s*(?:호|号)?",
            r"(?:제\s*)?(\d+)\s*(?:호|号)\s*(?:신문|news)"
        ]
        for p in patterns:
            m=re.search(p,u,re.I)
            if m:return int(m.group(1))
        return None

    @staticmethod
    def _asks_quiz(u):
        return any(x in u.lower() for x in ("퀴즈","quiz","クイズ","测验","cuestionario","question"))

    @staticmethod
    def _asks_stock(u):
        return any(x in u.lower() for x in ("주식","증권","stock","stocks","股票","株","돌돌증권"))

    @staticmethod
    def _asks_news(u):
        return any(x in u.lower() for x in ("돌이신문","신문","newspaper","news","新聞","报纸"))

    def _site_answer(self,u,user_id=None,access_token=None):
        site=self.site.with_token(access_token)
        n=self._news_number(u)
        if n is not None:
            row=site.news(n,user_id)
            if row and row.get("denied"):
                return f"그 신문은 아직 읽을 수 없어. 🐶 현재 네가 읽을 수 있는 돌이신문은 {row.get('max_readable',0)}호까지야."
            if not row:return "그 번호의 돌이신문은 현재 존재하지 않아."
            out=[f"📰 돌이신문 제{row['news_number']}호",row.get("rockey_news","")]
            if self._asks_quiz(u):
                q=row.get("question")
                if q:
                    out += ["","❓ 퀴즈",q]
                    choices=[row.get(f"choice{i}") for i in range(1,6) if row.get(f"choice{i}")]
                    if choices: out.append("\n".join(f"{i}. {v}" for i,v in enumerate(choices,1)))
            return "\n".join(x for x in out if x)

        if self._asks_news(u):
            limit=site.readable_news_limit(user_id)
            if limit<=0:return "로그인한 사용자가 읽을 수 있는 돌이신문 정보를 확인하지 못했어. 로그인 상태를 확인해줘. 🐶"
            rows=site.news(limit,user_id)
            if rows and not rows.get("denied"):
                if self._asks_quiz(u):
                    out=[f"❓ 현재 읽을 수 있는 가장 최신 퀴즈 — 돌이신문 제{limit}호",rows.get("question","")]
                    choices=[rows.get(f"choice{i}") for i in range(1,6) if rows.get(f"choice{i}")]
                    if choices:out.append("\n".join(f"{i}. {v}" for i,v in enumerate(choices,1)))
                    return "\n".join(x for x in out if x)
                return f"📰 현재 읽을 수 있는 가장 최신 돌이신문은 제{limit}호야.\n\n{rows.get('rockey_news','')}"
            return "현재 읽을 수 있는 돌이신문을 찾지 못했어."

        if self._asks_stock(u):
            rows=site.stock()
            if not rows:return "현재 돌돌증권 데이터를 불러오지 못했어. 잠시 후 다시 시도해줘."
            out=["📈 현재 돌돌증권 상황"]
            for x in rows:
                try: price=f"{int(x.get('current_price') or 0):,}"
                except Exception: price=str(x.get('current_price') or '-')
                out.append(f"• {x.get('name','')} ({x.get('ticker','')}): {price} 돌돌코인 | 변동 {x.get('change_amount',0):+,} ({x.get('change_percent',0)}%) | 위험도 {x.get('risk_label','미지정')}")
            return "\n".join(out)
        return None

    @staticmethod
    def _needs_web(u):
        return any(x in u.lower() for x in ("최신","현재","오늘","어제","내일","최근","실시간","지금","뉴스","날씨","환율","주가","가격","업데이트","latest","current","today","news","weather","price","最新","現在","今日"))

    def reply(self,user,user_id=None,access_token=None):
        u=normalize_query(user)
        lang=detect(u)
        key=re.sub(r"[?!？！，,\.\s]+$","",u).lower()
        dialogue=self._dialogue(user_id)

        if key in self.smalltalk:
            ans=self.smalltalk[key]
        else:
            ans=self._site_answer(u,user_id,access_token)
            if ans is None:
                ans=self.kb.answer(u,threshold=.66)
            if ans is None and self.web_enabled and self._needs_web(u):
                ans=format_results(u,web_search(u,limit=5,timeout=4))
            if ans is None:
                prompt=dialogue.build_prompt(u)
                temp=.48 if lang in ("ko","en","ja","zh") else .55
                ans=generate(prompt,self.tok,self.model,100,temp,12,.82)
            if self._bad(ans):
                ans="내가 확실하게 확인할 수 있는 자료가 부족해서 추측하지 않을게. 질문을 조금 더 구체적으로 말해줘. 🐶"

        dialogue.add_turn(u,ans)
        return ans
