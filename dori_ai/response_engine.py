import re
from .retrieval import LocalKnowledge
from .memory import MemoryStore
from .dialogue import DialogueManager
from generate_final import generate

class ResponseEngine:
    """Fast local-first Dori response pipeline.

    1) exact/semantic local knowledge for factual accuracy
    2) common conversation intents for natural quick replies
    3) trained Transformer for unseen prompts
    4) quality gate prevents repetitive/garbled model output
    """
    def __init__(self,tok,model):
        self.tok,self.model=tok,model; self.memory=MemoryStore(); self.dialogue=DialogueManager(self.memory); self.kb=LocalKnowledge()
        self.smalltalk={
          '안녕':'안녕! 나는 돌이야 🐶 오늘은 무슨 이야기를 해볼까?',
          '안녕하세요':'안녕! 나는 돌이야 🐶 무엇을 도와줄까?',
          '고마워':'천만에! 도움이 되었다면 기뻐! 🐶',
          '고마워!':'천만에! 언제든 이야기해줘!',
          '뭐해':'지금 네 이야기를 듣고 있어!',
          '뭐 하고 있어':'지금 네 이야기를 듣고 있어!',
          '잘 지냈어':'응! 여기에서 기다리고 있었어. 🐶',
        }
    @staticmethod
    def _bad(text):
        if not text or len(text)<2:return True
        t=text.strip(); chars=list(t)
        if len(chars)>=12:
            # Reject obvious repetition loops common in under-trained tiny LMs.
            for n in (1,2,3,4,5):
                if len(chars)>=n*5 and len(set(tuple(chars[i:i+n]) for i in range(0,len(chars)-n+1,n)))<=2:return True
        if re.search(r'(.)\1{7,}',t): return True
        if any(x in t for x in ('<system>','<user>','<dori>','<eos>')): return True
        # A tiny under-trained model can loop on short phrases. Reject obvious loops.
        words=re.findall(r'[가-힣A-Za-z0-9]+',t)
        if len(words)>=10 and len(set(words))/len(words) < 0.55:
            return True
        if len(words)>=8:
            for n in (1,2,3):
                if len(words)>=n*4 and all(words[i:i+n]==words[:n] for i in range(0,min(len(words),n*4),n)):
                    return True
        if t.count('s')>max(6,len(t)//3): return True
        return False
    def reply(self,user):
        u=user.strip(); key=re.sub(r'[?!？！，,\.\s]+$','',u)
        if key in self.smalltalk: ans=self.smalltalk[key]
        else:
            ans=self.kb.answer(u,threshold=.58)
            if ans is None:
                prompt=self.dialogue.build_prompt(u)+'\n<dori>'
                ans=generate(prompt,self.tok,self.model,80,.55,12,.88)
                if self._bad(ans):
                    ans='그 질문은 내가 확실하게 알고 있는 범위를 벗어나 있어. 추측해서 틀리게 말하지 않고, 아는 내용부터 정확하게 설명해줄게.'
        self.dialogue.add_turn(u,ans); return ans
