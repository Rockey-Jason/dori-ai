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
        self.memory = MemoryStore(); self.dialogue = DialogueManager(self.memory)
        self.kb = LocalKnowledge(); self.site = SiteData()
        self.web_enabled = os.getenv("DORI_WEB_SEARCH", "1").lower() not in ("0", "false", "off")
        self.smalltalk = {
            "안녕":"안녕! 나는 돌이야 🐶 오늘은 무슨 이야기를 해볼까?", "안녕하세요":"안녕! 나는 돌이야 🐶 무엇을 도와줄까?",
            "고마워":"천만에! 도움이 되었다면 기뻐! 🐶", "hello":"Hello! I'm Dori AI. 🐶", "hi":"Hi! I'm Dori AI. 🐶",
            "hola":"¡Hola! Soy Dori AI. 🐶", "bonjour":"Bonjour ! Je suis Dori AI. 🐶", "こんにちは":"こんにちは！Dori AIだよ。🐶",
            "你好":"你好！我是Dori AI。🐶", "안녕!":"안녕! 나는 돌이야 🐶"
        }

    @staticmethod
    def _bad(text):
        if not text or len(text) < 2: return True
        t = text.strip()
        if any(x in t for x in ("<system>", "<user>", "<dori>", "<eos>")): return True
        if re.search(r"(.)\1{7,}", t): return True
        words = re.findall(r"[가-힣A-Za-z0-9]+", t)
        return len(words) >= 10 and len(set(words)) / len(words) < .55

    @staticmethod
    def _news_number(u):
        m = re.search(r"(?:돌이\s*신문|신문|newspaper|news|新聞|报纸)\s*(?:제\s*)?(\d+)\s*(?:호|号)?", u, re.I)
        if m: return int(m.group(1))
        m = re.search(r"(?:제\s*)?(\d+)\s*(?:호|号)\s*(?:신문|news)", u, re.I)
        return int(m.group(1)) if m else None

    @staticmethod
    def _asks_quiz(u): return any(x in u.lower() for x in ("퀴즈", "quiz", "クイズ", "测验", "cuestionario"))
    @staticmethod
    def _asks_stock(u): return any(x in u.lower() for x in ("주식", "증권", "stock", "stocks", "股票", "株", "돌돌증권"))

    def _site_answer(self, u, user_id=None):
        n = self._news_number(u)
        if n is not None:
            row = self.site.news(n, user_id)
            if row and row.get("denied"):
                return f"그 신문은 아직 읽을 수 없어. 🐶 현재 네가 읽을 수 있는 돌이신문은 **{row.get('max_readable', 0)}호까지**야."
            if not row: return "그 번호의 돌이신문을 찾지 못했어."
            out = [f"📰 돌이신문 제{row['news_number']}호", row.get("rockey_news", "")]
            if self._asks_quiz(u) or any(k in u for k in ("문제", "question", "quiz")):
                out += ["", "❓ 퀴즈", row.get("question", "")]
                choices = [row.get(f"choice{i}") for i in range(1, 6) if row.get(f"choice{i}")]
                if choices: out.append("\n".join(f"{i}. {v}" for i, v in enumerate(choices, 1)))
            return "\n".join(x for x in out if x)
        if self._asks_stock(u):
            rows = self.site.stock()
            if not rows: return None
            out = ["📈 현재 돌돌증권 상황"]
            for x in rows:
                out.append(f"• {x['name']} ({x['ticker']}): {x['current_price']:,} 돌돌코인 | 변동 {x['change_amount']:+,} ({x['change_percent']}%) | 위험도 {x['risk_label']}")
            return "\n".join(out)
        return None

    @staticmethod
    def _needs_web(u):
        return any(x in u for x in ("최신", "현재", "오늘", "어제", "내일", "최근", "실시간", "지금", "뉴스", "날씨", "환율", "주가", "가격", "업데이트", "latest", "current", "today", "news", "weather", "price", "最新", "現在", "今日"))

    def reply(self, user, user_id=None):
        u = normalize_query(user); detect(u)
        key = re.sub(r"[?!？！，,\.\s]+$", "", u)
        if key in self.smalltalk: ans = self.smalltalk[key]
        else:
            ans = self._site_answer(u, user_id)
            if ans is None: ans = self.kb.answer(u, threshold=.58)
            if ans is None and self.web_enabled and self._needs_web(u):
                ans = format_results(u, web_search(u, limit=5, timeout=4))
            if ans is None:
                ans = generate(self.dialogue.build_prompt(u), self.tok, self.model, 120, .62, 16, .90)
            if self._bad(ans):
                ans = "그 질문은 내가 확실하게 확인할 수 있는 자료가 부족해. 추측해서 틀리게 말하지 않을게. 질문을 조금 더 구체적으로 해줘. 🐶"
        self.dialogue.add_turn(u, ans)
        return ans
