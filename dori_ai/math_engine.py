"""Safe, dependency-light mathematics engine for Dori AI.
No eval/exec and no external AI model are used here.
"""
import ast, math, operator, re
from decimal import Decimal, getcontext
getcontext().prec = 40

_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log10": math.log10,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
}
_CONST = {"pi": math.pi, "e": math.e, "tau": math.tau}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}

class MathError(ValueError): pass

def _eval(node):
    if isinstance(node, ast.Expression): return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int,float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        v=_eval(node.operand); return v if isinstance(node.op,ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a,b=_eval(node.left),_eval(node.right)
        if type(node.op) is ast.Pow and abs(b)>1000: raise MathError("지수가 너무 커")
        return _OPS[type(node.op)](a,b)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords: raise MathError("지원하지 않는 인자")
        vals=[_eval(x) for x in node.args]
        return _FUNCS[node.func.id](*vals)
    if isinstance(node, ast.Name) and node.id in _CONST: return _CONST[node.id]
    raise MathError("지원하지 않는 수식")

def calculate(expr):
    s=str(expr).strip().replace("×","*").replace("÷","/").replace("−","-").replace("^","**")
    s=re.sub(r"(?i)(계산해줘|계산|값을 구해줘|풀어줘|답은\??)$", "", s).strip()
    if not s or len(s)>240: raise MathError("수식이 비어 있거나 너무 길어")
    # Percentage: 15% -> 0.15
    s=re.sub(r"(?<![A-Za-z0-9_.])([0-9]+(?:\.[0-9]+)?)\s*%", r"(\1/100)", s)
    try:
        tree=ast.parse(s, mode="eval")
        v=float(_eval(tree))
    except Exception as e:
        raise MathError(str(e))
    if not math.isfinite(v): raise MathError("유한한 결과가 아니야")
    return v

def _fmt(v):
    if abs(v-round(v))<1e-12: return f"{int(round(v)):,}"
    return f"{v:,.12g}"

def solve_linear(eq):
    s=str(eq).strip().replace("−","-").replace("×","*").replace(" ","")
    if "=" not in s or not re.search(r"x", s, re.I): return None
    left,right=s.split("=",1)
    def coeff(side):
        side=re.sub(r"(?i)(\d+(?:\.\d+)?)x", r"\1*x", side)
        side=re.sub(r"(?i)(?<![\w.*])x", "1*x", side)
        side=side.replace("-","+-")
        if side.startswith("+"): side=side[1:]
        a=b=0.0
        for part in side.split("+"):
            if not part: continue
            if "*x" in part.lower():
                c=part.lower().split("*x",1)[0]
                a += float(c or "1")
            else:
                b += calculate(part)
        return a,b
    try:
        a,b=coeff(left); c,d=coeff(right); a-=c; b-=d
        if abs(a)<1e-12: return "해가 무수히 많아." if abs(b)<1e-12 else "해가 없어."
        return f"x = {_fmt(-b/a)}"
    except Exception:
        return None

def answer(text, lang="ko"):
    q=str(text).strip()
    equation=solve_linear(q)
    if equation:
        if lang=="en": return f"Solution: {equation}"
        if lang=="ja": return f"解: {equation}"
        if lang=="zh": return f"解：{equation}"
        return f"🧮 {equation}"
    # Strip common math lead-ins.
    candidate=re.sub(r"(?i)^(계산해줘|계산해 줘|calculate|what is|solve|답을 구해줘|답은|계산)\s*[:：]?\s*", "", q).strip()
    # Natural Korean arithmetic.
    candidate=re.sub(r"\s*더하기\s*", "+", candidate)
    candidate=re.sub(r"\s*빼기\s*", "-", candidate)
    candidate=re.sub(r"\s*(?:곱하기|곱셈)\s*", "*", candidate)
    candidate=re.sub(r"\s*(?:나누기|나눗셈)\s*", "/", candidate)
    candidate=re.sub(r"\s*(?:는|은)\s*(?:얼마야|얼마입니까|몇이야|몇인가요)\s*\??$", "", candidate).strip()
    if not re.search(r"[0-9]", candidate): return None
    if not re.search(r"[+\-*/%^×÷()]|sqrt|sin|cos|tan|log|pi|\be\b", candidate, re.I): return None
    try: v=calculate(candidate)
    except MathError: return None
    f=_fmt(v)
    if lang=="en": return f"🧮 Answer: {f}"
    if lang=="ja": return f"🧮 答え：{f}"
    if lang=="zh": return f"🧮 答案：{f}"
    return f"🧮 답: {f}"
