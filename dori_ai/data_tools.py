"""Deterministic corpus expansion utilities used by the training pipeline."""
from pathlib import Path
import json, math, random

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"data"/"generated"/"foundation_math.jsonl"

def build_math_dataset(count=25000, seed=20260924):
    rng=random.Random(seed); OUT.parent.mkdir(parents=True,exist_ok=True)
    rows=[]; seen=set()
    templates=[
        ("{a} + {b}", lambda a,b:a+b),("{a} - {b}",lambda a,b:a-b),
        ("{a} × {b}",lambda a,b:a*b),("{a} ÷ {b}",lambda a,b:a/b),
        ("{a}² + {b}²",lambda a,b:a*a+b*b),
        ("{a}의 {p}%",lambda a,p:a*p/100),
    ]
    while len(rows)<count:
        a=rng.randint(-999,999); b=rng.randint(1,999); p=rng.randint(1,99)
        kind=rng.randrange(len(templates)); tpl,fn=templates[kind]
        try:
            if kind==5: q=tpl.format(a=a,p=p); v=fn(a,p)
            else: q=tpl.format(a=a,b=b); v=fn(a,b)
            ans=f"{v:.12g}"
        except Exception: continue
        variants=[f"계산해줘: {q}",f"What is {q}?",f"계산: {q}"]
        for q2 in variants:
            key=(q2,ans)
            if key not in seen:
                seen.add(key); rows.append({"question":q2,"answer":f"{ans}"})
                if len(rows)>=count: break
    OUT.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n",encoding="utf-8")
    return len(rows)

if __name__=="__main__": print(build_math_dataset())
