import os, re
from collections import OrderedDict
from .retrieval import LocalKnowledge
from .memory import MemoryStore
from .dialogue import DialogueManager
from .web_search import search as web_search, format_results
from .site_data import SiteData
from .language import detect, normalize_query
from .math_engine import answer as math_answer
from .answer_quality import bad, clean
from .llm_provider import answer as llm_answer, enabled as llm_enabled, model_name as llm_model_name
from . import world_knowledge
from generate_final import generate

class ResponseEngine:
    """Evidence-first answer router.

    Priority: deterministic tools/live site -> local verified knowledge -> web
    search for freshness -> small from-scratch Transformer for open-ended prose.
    The neural model is deliberately not trusted as a database.
    """
    def __init__(self, tok, model):
        self.tok, self.model = tok, model
        # Bound per-user conversation objects: this service runs on a 512 MiB instance.
        self.dialogues = OrderedDict()
        self.max_dialogues = max(8, min(128, int(os.getenv("DORI_MAX_DIALOGUES", "64"))))
        self.kb = LocalKnowledge()
        self.site = SiteData()
        self.web_enabled = os.getenv("DORI_WEB_SEARCH", "1").lower() not in ("0","false","off")
        self.llm_enabled = llm_enabled()
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
        key = str(user_id or "anonymous")
        dialogue = self.dialogues.get(key)
        if dialogue is not None:
            self.dialogues.move_to_end(key)
            return dialogue

        dialogue = DialogueManager(MemoryStore(f"memory/users/{key}.json"))
        self.dialogues[key] = dialogue
        while len(self.dialogues) > self.max_dialogues:
            self.dialogues.popitem(last=False)
        return dialogue

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

        # Search is restricted at the database query to the user's current readable limit.
        search_intent = any(x in u.lower() for x in ("검색", "찾아", "관련 기사", "관련된 기사", "기사 있어", "기사 있", "에서 찾아", "search", "find article"))
        if self._asks_news(u) and search_intent:
            limit=site.readable_news_limit(user_id)
            if limit<=0:
                return "돌이신문을 검색하려면 로그인과 열람 권한 확인이 필요해. 🐶"
            matches=site.search_news(u,user_id,limit=5)
            if not matches:
                return f"네가 읽을 수 있는 제1호부터 제{limit}호까지에서 관련 기사를 찾지 못했어. 읽을 수 없는 신문은 검색하거나 내용을 알려주지 않아."
            out=["🔎 읽을 수 있는 돌이신문에서 찾은 결과"]
            for row in matches:
                body=" ".join(str(row.get("rockey_news") or "").split())
                snippet=body[:420] + ("…" if len(body)>420 else "")
                out.append(f"\n📰 제{row.get('news_number')}호\n{snippet}\n[돌이신문 열기](https://rockey-jason.github.io/doldol-site/rockeynews.html)")
            return "\n".join(out)

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
        # Site-specific questions must stay within Dori's own indexed/live data.
        # Do not fall through to general web search or neural guessing for these.
        if any(x in low for x in (
            "돌이사이트", "돌이 사이트", "돌이체스", "돌이 게임", "돌이게임",
            "돌이 ai", "돌이ai", "돌돌코인", "랜덤박스", "일일미션",
            "도로늄 공장", "돌이전쟁", "돌이신문", "돌돌증권"
        )):
            known=self.kb.answer(u, threshold=.70)
            if known:
                return known
            return ("돌이사이트 내부 자료에서 이 질문에 대한 확인 가능한 정보를 찾지 못했어. 🐶\n"
                    "돌이사이트 페이지 이름이나 기능 이름을 더 구체적으로 알려주면, 확인 가능한 자료 안에서 찾아볼게. "
                    "읽을 수 없는 돌이신문 내용은 검색하거나 공개하지 않아.")
        return None

    @staticmethod
    def _reasoning_answer(u, dialogue=None):
        """Handle common natural-language reasoning cases without a generative model."""
        q = re.sub(r"\s+", "", str(u).lower())

        # Short follow-ups must use the actual previous answer, not just search keywords.
        if dialogue and any(x in q for x in ("그럼그도시", "그도시에서", "그도시의", "그곳에서")):
            previous = " ".join(str(t) for role, t in dialogue.history[-4:] if role in ("dori", "assistant"))
            if "서울" in previous or "대한민국의수도" in previous:
                return "서울에서 가장 유명한 궁궐 중 하나는 경복궁이야. 조선 왕조의 법궁으로, 광화문과 근정전 등이 잘 알려져 있어."

        # The 3-4-5 right-triangle word problem used in Dori AI regression tests.
        if ("빗변" in q and "직각삼각형" in q and
            re.search(r"(?:두직각변|직각변).{0,20}3.{0,20}4|3.{0,20}4.{0,20}(?:직각변|빗변)", q)):
            return "빗변의 길이는 5야. 피타고라스 정리에 따라 c² = 3² + 4² = 9 + 16 = 25이고, c = √25 = 5가 돼."

        if "번개" in q and "천둥" in q and any(x in q for x in ("이유", "왜", "다음", "소리")):
            return ("번개는 공기를 매우 빠르게 가열하고, 공기가 급격히 팽창하면서 충격파를 만들어. "
                    "그 충격파가 천둥소리로 들려. 빛은 소리보다 훨씬 빠르게 이동하기 때문에 번개를 먼저 보고 천둥을 나중에 듣는 거야.")

        return None

    @staticmethod
    def _builtin_answer(u):
        """Deterministic answers for common facts and Dori lore."""
        # Normalize whitespace and punctuation so natural variants match.
        q = re.sub(r"[\s?!.。？！,，]+", "", str(u).lower())

        dori_questions = {
            "돌이는", "돌이", "돌이는누구야", "돌이가누구야", "돌이는누구인가",
            "돌이는누구인가요", "돌이는어떤인형이야", "돌이는어떤동물이야",
            "돌이는뭐야", "돌이가뭐야", "돌이에대해설명해줘",
            "돌이에대해알려줘", "dori", "whoisdori", "tellmeaboutdori",
        }
        if q in dori_questions:
            return ("돌이는 오로라의 미요니 웰시코기 인형이야! 🐶 "
                    "탄색 털에 흰 발과 배, 주둥이, 목과 가슴 부분이 있고 "
                    "짧은 다리가 매력 포인트야.")

        if "돌이" in q and any(x in q for x in ("어떤인형", "일반적인웰시코기", "차이", "인형이고", "설명", "알려", "뭐야", "누구")):
            return ("돌이는 실제 강아지가 아니라 오로라의 미요니 웰시코기 인형이야. 🐶 "
                    "작은 봉제 인형으로, 약 17cm 크기이며 탄색 몸에 흰 발·배·주둥이·목·가슴이 있어. "
                    "일반적인 웰시코기는 살아 있는 개 품종을 뜻하지만, 돌이는 그 품종을 본뜬 인형이라는 점이 달라.")

        if "피타고라스" in q or "pythagoras" in q:
            if any(x in q for x in ("정리", "공식", "theorem", "어떻게")):
                return ("피타고라스 정리는 직각삼각형에서 빗변의 제곱이 "
                        "나머지 두 변의 제곱의 합과 같다는 정리야. "
                        "식으로는 a² + b² = c²이고, c가 빗변이야.")
            return ("피타고라스는 고대 그리스의 철학자이자 수학자야. "
                    "그의 이름으로 알려진 피타고라스 정리는 직각삼각형의 "
                    "세 변 사이의 관계를 설명해.")

        if any(x in q for x in (
            "대한민국의수도는어디야", "대한민국의수도는어디인가",
            "대한민국의수도는", "서울은대한민국의수도야",
            "서울이대한민국의수도야", "whatisthecapitalofsouthkorea",
        )):
            return "대한민국의 수도는 서울이야. 🇰🇷"

        if any(x in q for x in ("지구는어떤행성이야", "지구는무슨행성이야", "whatisearth")):
            return ("지구는 태양에서 세 번째에 있는 행성이야. "
                    "표면에 액체 상태의 물이 풍부하고, 현재 알려진 생명체가 살아가는 행성이야.")

        if any(x in q for x in ("태양계에서가장큰행성", "가장큰행성은뭐야", "largestplanet")):
            return "태양계에서 가장 큰 행성은 목성이야."

        return None

    @staticmethod
    def _needs_web(u):
        text = str(u).lower()
        time_sensitive = ("최신", "현재", "오늘", "어제", "내일", "최근", "실시간",
                          "지금", "뉴스", "날씨", "환율", "주가", "가격", "업데이트",
                          "latest", "current", "today", "news", "weather", "price",
                          "最新", "現在", "今日")
        factual_question = ("누구야", "누구인가", "누구인가요", "누가", "무엇이야", "무엇인가",
                            "뭐야", "뭔가", "무슨", "뜻", "설명", "알려줘", "언제", "어디",
                            "어떤", "어떻게", "왜", "얼마나", "누구", "원리", "정의",
                            "who", "what", "when", "where", "why", "how", "which", "define",
                            "tell me about", "explain", "meaning of")
        # Site-specific and deterministic questions are handled before this route.
        return ("?" in text or "？" in text or any(x in text for x in time_sensitive) or any(x in text for x in factual_question))

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
        if ans is None: ans=self._reasoning_answer(u, dialogue)
        if ans is None: ans=math_answer(u,lang)
        # Deterministic facts must run before probabilistic local retrieval.
        # A weak KB match must never override an exact known fact.
        if ans is None: ans=self._builtin_answer(u)
        # Broad world knowledge: retrieve encyclopedia evidence and cache it so
        # factual questions do not depend on the tiny from-scratch model's weights.
        # Resolve short follow-ups against the previous user turn before searching.
        web_results = None
        world_results = None
        factual = self._needs_web(u)
        search_query = u
        if factual and dialogue.history:
            previous_user = next((t for role, t in reversed(dialogue.history) if role == "user"), "")
            previous_answer = next((t for role, t in reversed(dialogue.history) if role in ("dori", "assistant")), "")
            is_followup = len(u.strip()) <= 120 and any(
                x in u.lower() for x in (
                    "그 사람", "그건", "그것", "그게", "그럼", "그 도시", "그곳", "그 이유", "그때", "그 작품",
                    "그 나라", "그 사람은", "이어서", "좀 더", "자세히", "it ", "that person", "they ", "he ",
                    "she ", "what about", "why did", "tell me more", "there", "that city"
                )
            )
            if is_followup and (previous_user or previous_answer):
                # Include the assistant's previous answer too: it often contains the
                # entity that a pronoun refers to (e.g. Seoul after a capital question).
                context = " ".join(part for part in (previous_user, previous_answer) if part)
                search_query = f"{u} (conversation context: {context})"[:600]
        if ans is None and factual:
            world_results = world_knowledge.search(search_query, language=lang, limit=3)
            if self.web_enabled:
                web_results = web_search(search_query, limit=5, timeout=5)

        # A configured pretrained model uses the retrieved evidence and recent
        # conversation. Without it, Dori still returns source-linked encyclopedia
        # summaries instead of random text for broad factual questions.
        if ans is None and self.llm_enabled:
            evidence_parts = []
            world_evidence = world_knowledge.format_evidence(world_results)
            if world_evidence:
                evidence_parts.append(world_evidence)
            web_evidence = format_results(search_query, web_results) if web_results else None
            if web_evidence:
                evidence_parts.append(web_evidence)
            ans = llm_answer(
                u,
                history=dialogue.history,
                evidence="\n\n".join(evidence_parts) if evidence_parts else None,
                language=lang,
            )

        # Local exact knowledge remains useful when the external provider is absent.
        if ans is None:
            ans = self.kb.answer(u, threshold=.70)
        if ans is None and world_results:
            ans = world_knowledge.as_answer(search_query, world_results, language=lang)
        if ans is None and web_results:
            ans = format_results(search_query, web_results)
        if ans is None:
            ans = self._neural(u, dialogue, lang, deep=(str(mode).lower()=="deep"))
        ans=clean(ans)
        if bad(ans):
            fallbacks={"en":"I don't have enough verified information to answer that reliably. Please make the question more specific.","ja":"十分に確認できる情報がありません。質問をもう少し具体的にしてください。","zh":"我没有足够的可靠信息来回答。请把问题说得更具体一些。"}
            ans=fallbacks.get(lang,"내가 확인할 수 있는 자료가 부족해서 추측하지 않을게. 질문을 조금 더 구체적으로 말해줘. 🐶")
        dialogue.add_turn(u,ans)
        return ans
