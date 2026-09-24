import os, re
from .retrieval import LocalKnowledge
from .memory import MemoryStore
from .dialogue import DialogueManager
from .web_search import search as web_search, format_results
from .site_data import SiteData
from .language import detect, normalize_query
from .math_engine import answer as math_answer
from .answer_quality import bad, clean
from generate_final import generate

class ResponseEngine:
    """Evidence-first answer router.

    Priority: deterministic tools/live site -> local verified knowledge -> web
    search for freshness -> small from-scratch Transformer for open-ended prose.
    The neural model is deliberately not trusted as a database.
    """
    def __init__(self, tok, model):
        self.tok, self.model = tok, model
        self.dialogues = {}
        self.kb = LocalKnowledge()
        self.site = SiteData()
        self.web_enabled = os.getenv("DORI_WEB_SEARCH", "1").lower() not in ("0","false","off")
        self.smalltalk = {
            "안녕":"안녕! 나는 돌이 AI야 🐶 무엇을 같이 알아볼까?",
            "안녕하세요":"안녕! 나는 돌이 AI야 🐶 질문을 정확하게 도와줄게!",
            "고마워":"천만에! 🐶",
            "hello":"Hello! I'm Dori AI. 🐶 How can I help?",
            "hi":"Hi! I'm Dori AI. 🐶",
            "hola":"¡Hola! Soy Dori AI. 🐶 ¿En qué puedo ayudarte?",
            "bonjour":"Bonjour ! Je suis Dori AI. 🐶",
            "hallo":"Hallo! Ich bin Dori AI. 🐶",
            "ciao":"Ciao! Sono Dori AI. 🐶",
            "こんにちは":"こんにちは！Dori AIだよ。🐶",
            "你好":"你好！我是Dori AI。🐶",
            "привет":"Привет! Я Dori AI. 🐶",
        }

    def _dialogue(self, user_id):
        key=str(user_id or "anonymous")
        if key not in self.dialogues:
            self.dialogues[key]=DialogueManager(MemoryStore(f"memory/users/{key}.json"))
        return self.dialogues[key]

    @staticmethod
    def _news_number(u):
        for p in [r"(?:돌이\s*신문|신문|newspaper|news|新聞|报纸)\s*(?:제\s*)?(\d+)\s*(?:호|号)?", r"(?:제\s*)?(\d+)\s*(?:호|号)\s*(?:신문|news)"]:
            m=re.search(p,u,re.I)
            if m:return int(m.group(1))
        return None
    @staticmethod
    def _asks_quiz(u): return any(x in u.lower() for x in ("퀴즈","quiz","クイズ","测验","cuestionario"))
    @staticmethod
    def _asks_stock(u): return any(x in u.lower() for x in ("주식","증권","stock","stocks","股票","株","돌돌증권"))
    @staticmethod
    def _asks_news(u): return any(x in u.lower() for x in ("돌이신문","신문","newspaper","news","新聞","报纸"))

    def _site_answer(self,u,user_id=None,access_token=None):
        site=self.site.with_token(access_token)
        n=self._news_number(u)
        if n is not None:
            row=site.news(n,user_id)
            if row and row.get("denied"):
                return f"그 신문은 아직 읽을 수 없어. 🐶 현재 네 계정에서 읽을 수 있는 돌이신문은 제{row.get('max_readable',0)}호까지야."
            if not row:return "그 번호의 돌이신문은 현재 존재하지 않아."
            out=[f"📰 돌이신문 제{row['news_number']}호",row.get('rockey_news','')]
            if self._asks_quiz(u) and row.get('question'):
                out += ["", "❓ 퀴즈", row['question']]
                choices=[row.get(f'choice{i}') for i in range(1,6) if row.get(f'choice{i}')]
                if choices: out.append("\n".join(f"{i}. {v}" for i,v in enumerate(choices,1)))
            return "\n".join(x for x in out if x)

        if self._asks_news(u):
            limit=site.readable_news_limit(user_id)
            if limit<=0:return "로그인한 사용자의 돌이신문 열람 권한을 확인하지 못했어. 로그인 상태를 확인해줘. 🐶"
            row=site.news(limit,user_id)
            if row and not row.get('denied'):
                if self._asks_quiz(u):
                    out=[f"❓ 현재 읽을 수 있는 최신 퀴즈 — 돌이신문 제{limit}호",row.get('question','')]
                    choices=[row.get(f'choice{i}') for i in range(1,6) if row.get(f'choice{i}')]
                    if choices:out.append("\n".join(f"{i}. {v}" for i,v in enumerate(choices,1)))
                    return "\n".join(x for x in out if x)
                return f"📰 현재 읽을 수 있는 최신 돌이신문은 제{limit}호야.\n\n{row.get('rockey_news','')}"
            return "현재 읽을 수 있는 돌이신문을 찾지 못했어."

        low=u.lower()
        if any(x in low for x in ("내 정보","내 등급","내 레벨","내 코인","내 경험치","my profile","my account")):
            profile=site.profile(user_id)
            if not profile:return "로그인한 사용자 정보를 확인하지 못했어. 로그인 상태를 확인해줘. 🐶"
            return (f"👤 {profile.get('name','사용자')}\n등급: Lv.{profile.get('user_level',0)} | 경험치 레벨: {profile.get('exp_level',0)}\n"
                    f"돌돌코인: {int(profile.get('doldolcoin') or 0):,}\n경험치: {int(profile.get('exp') or 0):,}\n"
                    f"칭호: {profile.get('equipped_title') or '없음'}\n읽을 수 있는 돌이신문: 제{profile.get('read_dori_news') or 0}호까지")

        if any(x in low for x in ("사이트 정보","돌이사이트 정보","site info","사이트에 뭐가 있어")):
            s=site.site_summary()
            return f"🌐 돌이사이트 정보\n돌이신문: {s['news_count']}개 (최신 제{s['latest_news']}호)\n활성 돌돌증권 종목: {s['stock_count']}개"

        if self._asks_stock(u):
            rows=site.stock()
            if not rows:return "현재 돌돌증권 데이터를 불러오지 못했어. 잠시 후 다시 시도해줘."
            q=low; specific=next((r for r in rows if str(r.get('name','')).lower() in q or str(r.get('ticker','')).lower() in q),None)
            if specific:
                x=specific
                return (f"📈 {x.get('name','')} ({x.get('ticker','')})\n현재가: {int(x.get('current_price') or 0):,} 돌돌코인\n"
                        f"변동: {x.get('change_amount',0):+,} ({x.get('change_percent',0)}%)\n위험도: {x.get('risk_label','미지정')}\n"
                        f"설명: {x.get('description','')}\n특징: {x.get('characteristics','')}\n잔여 수량: {x.get('available_shares','-')}")
            out=["📈 현재 돌돌증권 데이터"]
            for x in rows:
                out.append(f"• {x.get('name','')} ({x.get('ticker','')}): {int(x.get('current_price') or 0):,} 돌돌코인 | {x.get('change_amount',0):+,} ({x.get('change_percent',0)}%) | 위험도 {x.get('risk_label','미지정')}")
            return "\n".join(out)
        return None

    @staticmethod
    def _builtin_answer(u):
        """
        Small deterministic fact layer for questions that should never depend on
        probabilistic Transformer generation. This also prevents slow/fragile
        neural inference on simple factual questions.
        """
        q=re.sub(r"\\s+","",u.lower())

        capital_patterns=(
            "대한민국의수도는어디야?",
            "대한민국의수도는어디인가?",
            "대한민국의수도는어디인가요?",
            "대한민국의수도는?",
            "대한민국의수도는서울이야?",
            "서울은대한민국의수도야?",
            "서울이대한민국의수도야?",
        )
        if q in capital_patterns:
            return "대한민국의 수도는 서울이야. 🇰🇷"

        return None

    @staticmethod
    def _needs_web(u):
        return any(x in u.lower() for x in ("최신","현재","오늘","어제","내일","최근","실시간","지금","뉴스","날씨","환율","주가","가격","업데이트","latest","current","today","news","weather","price","最新","現在","今日"))

    def _neural(self,u,dialogue,lang,deep=False):
        prompt=dialogue.build_prompt(u)
        # Lower temperature is intentional: this tiny from-scratch model is not a
        # substitute for a large pretrained reasoning model.
        temp=.34 if deep else .42
        # Keep public fast-mode inference short enough for small CPU instances.
        # Deterministic/site/KB answers are handled before this point.
        return generate(prompt,self.tok,self.model,80 if deep else 48,temp,12,.82)

    def reply(self,user,user_id=None,access_token=None,mode="fast"):
        u=normalize_query(user); lang=detect(u)
        key=re.sub(r"[?!？！，,.\s]+$","",u).lower()
        dialogue=self._dialogue(user_id)
        ans=self.smalltalk.get(key)
        if ans is None: ans=self._site_answer(u,user_id,access_token)
        if ans is None: ans=math_answer(u,lang)
        # Deterministic facts must run before probabilistic local retrieval.
        # A weak KB match must never override an exact known fact.
        if ans is None: ans=self._builtin_answer(u)
        if ans is None: ans=self.kb.answer(u,threshold=.70)
        if ans is None and self.web_enabled and self._needs_web(u):
            results=web_search(u,limit=6,timeout=5)
            ans=format_results(u,results) if results else None
        if ans is None: ans=self._neural(u,dialogue,lang,deep=(str(mode).lower()=="deep"))
        ans=clean(ans)
        if bad(ans):
            fallbacks={"en":"I don't have enough verified information to answer that reliably. Please make the question more specific.","ja":"十分に確認できる情報がありません。質問をもう少し具体的にしてください。","zh":"我没有足够的可靠信息来回答。请把问题说得更具体一些。"}
            ans=fallbacks.get(lang,"내가 확인할 수 있는 자료가 부족해서 추측하지 않을게. 질문을 조금 더 구체적으로 말해줘. 🐶")
        dialogue.add_turn(u,ans)
        return ans
