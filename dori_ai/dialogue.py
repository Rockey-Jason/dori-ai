class DialogueManager:
    def __init__(self,memory): self.memory=memory; self.history=[]
    def build_prompt(self,user_text):
        mem=self.memory.context(user_text,5)
        recent=self.history[-6:]
        lines=['<system>너는 Dori AI다. 친절하고 정확하게 답한다. 모르면 추측하지 않는다.']
        if mem: lines.append('관련 기억:\n'+mem)
        for role,text in recent: lines.append(f'<{role}>{text}')
        lines.append('<user>'+user_text); lines.append('<dori>'); return '\n'.join(lines)
    def add_turn(self,user,assistant): self.history += [('user',user),('dori',assistant)]
