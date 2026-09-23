import re
import os
from .retrieval import LocalKnowledge
from .memory import MemoryStore
from .dialogue import DialogueManager
from .web_search import search as web_search, format_results
from generate_final import generate

class ResponseEngine:
    """Dori AI v2.3 response pipeline.

    Priority:
      1. deterministic small-talk
      2. verified local knowledge
      3. optional public-web search for unknown/current questions
      4. from-scratch Transformer generation
      5. quality gate
    """
    def __init__(self,tok,model):
        self.tok,self.model=tok,model
        self.memory=MemoryStore()
        self.dialogue=DialogueManager(self.memory)
        self.kb=LocalKnowledge()
        self.web_enabled=os.getenv("DORI_WEB_SEARCH","1").lower() not in ("0","false","off")
        self.smalltalk={
          "안녕":"안녕! 나는 돌이야 🐶 오늘은 무슨 이야기를 해볼까?",
          "안녕하세요":"안녕! 나는 돌이야 🐶 무엇을 도와줄까?",
          "고마워":"천만에! 도움이 되었다면 기뻐! 🐶",
          "고마워!":"천만에! 언제든 이야기해줘!",
          "뭐해":"지금 네 이야기를 듣고 있어!",
          "뭐 하고 있어":"지금 네 이야기를 듣고 있어!",
          "잘 지냈어":"응! 여기에서 기다리고 있었어. 🐶",
        }

    @staticmethod
    def _bad(text):
        if not text or len(text)<2:return True
        t=text.strip()
        if any(x in t for x in ("<system>","<user>","<dori>","<eos>")): return True
        if re.search(r"(.)\1{7,}",t): return True
        words=re.findall(r"[가-힣A-Za-z0-9]+",t)
        if len(words)>=10 and len(set(words))/len(words)<.55:return True
        if t.count("s")>max(6,len(t)//3):return True
        return False

    @staticmethod
    def _needs_web(u):
        # Current/fresh questions should not rely on a tiny offline model.
        current=("최신","현재","오늘","어제","내일","최근","실시간","지금","뉴스","날씨","환율","주가","가격","업데이트")
        if any(x in u for x in current): return True
        # Search general factual questions when local knowledge has no strong hit.
        patterns=("누구야","뭐야","무엇","무슨 뜻","언제","어디","왜","어떻게","설명","알려줘","비교","차이")
        return any(x in u for x in patterns)

    def _web_answer(self,u):
        if not self.web_enabled or not self._needs_web(u): return None
        results=web_search(u,limit=5,timeout=4)
        return format_results(u,results) if results else None

    def reply(self,user):
        u=user.strip()
        key=re.sub(r"[?!？！，,\.\s]+$","",u)
        if key in self.smalltalk:
            ans=self.smalltalk[key]
        else:
            ans=self.kb.answer(u,threshold=.58)
            if ans is None:
                # Web search is a source-finding layer, not an AI model.
                ans=self._web_answer(u)
            if ans is None:
                prompt=self.dialogue.build_prompt(u)
                ans=generate(prompt,self.tok,self.model,100,.62,16,.90)
                if self._bad(ans):
                    ans="그 질문은 내가 확실하게 알고 있는 범위를 벗어나 있어. 추측해서 틀리게 말하지 않을게. 질문을 조금 더 구체적으로 알려주면 내가 가진 자료를 기준으로 다시 확인해볼게."
        self.dialogue.add_turn(u,ans)
        return ans
