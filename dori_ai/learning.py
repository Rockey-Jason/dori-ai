import json, os, threading, time
from pathlib import Path
import numpy as np
from .bpe_tokenizer import BPETokenizer
from .core.optimizer import Adam
from .core.transformer import load_model
from .site_data import SiteData

ROOT = Path(__file__).resolve().parent.parent
CP = ROOT / "checkpoints" / "best.npz"
MP = Path(str(CP) + ".json")
STATUS = ROOT / "runtime" / "learning_status.json"
AUTO = ROOT / "data" / "learning" / "auto_corpus.txt"

class LearningManager:
    def __init__(self, reload_callback=None):
        self.reload_callback = reload_callback
        self.lock = threading.Lock()
        self.running = False
        self.stop_requested = False
        self.status = {"running":False,"phase":"idle","progress":0,"message":"학습 대기 중","examples":0,"tokens":0,"step":0,"steps":0,"loss":None,"started_at":None,"finished_at":None,"error":None}
        self._load()

    def _load(self):
        try:
            if STATUS.exists(): self.status.update(json.loads(STATUS.read_text(encoding="utf-8")))
        except Exception: pass

    def _save(self):
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(self.status, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_status(self):
        with self.lock: return dict(self.status)

    def _set(self, **kw):
        with self.lock:
            self.status.update(kw); self._save()

    def _stop(self):
        with self.lock: return self.stop_requested

    def stop(self):
        with self.lock:
            if not self.running: return False
            self.stop_requested = True
            self.status["message"] = "학습 중지 요청을 처리하는 중..."
            self._save()
            return True

    def start(self, steps=None):
        with self.lock:
            if self.running: return False, "이미 학습 중이야."
            n = int(steps or os.getenv("DORI_TRAIN_STEPS","600"))
            self.running = True; self.stop_requested = False
            self.status.update({"running":True,"phase":"starting","progress":0,"message":"학습 모드를 시작하는 중...","examples":0,"tokens":0,"step":0,"steps":n,"loss":None,"started_at":time.time(),"finished_at":None,"error":None})
            self._save()
            threading.Thread(target=self._run, daemon=True, name="dori-learning").start()
            return True, "학습 모드를 시작했어."

    @staticmethod
    def _add(out, line):
        line = " ".join(str(line).split())
        if len(line) >= 20: out.append(line)

    def collect(self):
        out = []
        cd = ROOT / "data" / "corpus"
        if cd.exists():
            for p in sorted(cd.glob("*.txt")):
                try:
                    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines(): self._add(out,line)
                except Exception: pass
        for p in sorted((ROOT/"data").glob("*.jsonl")):
            try:
                for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    try: row=json.loads(raw)
                    except Exception: continue
                    q=row.get("question") or row.get("input") or row.get("prompt")
                    a=row.get("answer") or row.get("output") or row.get("response")
                    if q and a: self._add(out,f"질문: {q} 답변: {a}")
            except Exception: pass
        site=SiteData()
        try:
            for row in site.public_news_all():
                n = row.get("news_number")
                self._add(out,f"돌이신문 제{n}호: {row.get('rockey_news','')}")
                if row.get("question"): self._add(out,f"돌이신문 퀴즈: {row.get('question','')}")
        except Exception: pass
        try:
            for row in site.stock():
                self._add(out,"돌돌증권 종목: "+f"{row.get('name','')} {row.get('ticker','')} 설명: {row.get('description','')} 특징: {row.get('characteristics','')} 위험도: {row.get('risk_label','')}")
        except Exception: pass
        seen=set(); merged=[]
        for line in out:
            k=line.casefold()
            if k not in seen: seen.add(k); merged.append(line)
        AUTO.parent.mkdir(parents=True, exist_ok=True)
        AUTO.write_text("\n".join(merged)+"\n",encoding="utf-8")
        self._set(phase="collected",progress=15,message=f"학습 자료 {len(merged):,}개를 준비했어.",examples=len(merged))
        return merged

    def _train(self, lines, steps):
        if not CP.exists() or not MP.exists(): raise RuntimeError("기존 체크포인트가 없어. 먼저 기본 학습을 완료해야 해.")
        meta=json.loads(MP.read_text(encoding="utf-8"))
        tok=BPETokenizer.load(ROOT/meta["tokenizer"])
        model=load_model(str(CP),meta["model_config"])
        texts=[tok.encode(x) for x in lines]
        texts=[x for x in texts if len(x)>=8]
        if not texts: raise RuntimeError("학습 가능한 텍스트가 충분하지 않아.")
        rng=np.random.default_rng(20260924)
        seq_len=min(int(meta["model_config"].get("max_len",64)),96)
        micro=max(1,int(os.getenv("DORI_TRAIN_MICROBATCH","4")))
        lr=float(os.getenv("DORI_TRAIN_LR","1.5e-4"))
        opt=Adam(model.parameters(),lr=lr,weight_decay=1e-5)
        self._set(phase="training",progress=20,message=f"Transformer 가중치를 학습 중... {sum(map(len,texts)):,} 토큰",tokens=sum(map(len,texts)))
        last=None
        for step in range(1,steps+1):
            if self._stop(): return None
            model.zero_grad(); losses=[]
            for _ in range(micro):
                ids=texts[int(rng.integers(0,len(texts)))]
                if len(ids)>seq_len:
                    s=int(rng.integers(0,len(ids)-seq_len+1)); ids=ids[s:s+seq_len]
                x=np.asarray(ids[:-1],dtype=np.int64); y=np.asarray(ids[1:],dtype=np.int64)
                logits=model(x)
                z=logits.data-np.max(logits.data,axis=1,keepdims=True)
                probs=np.exp(z); probs/=np.sum(probs,axis=1,keepdims=True)
                idx=np.arange(len(y)); loss=-np.log(np.maximum(probs[idx,y],1e-12)).mean(); losses.append(float(loss))
                grad=probs; grad[idx,y]-=1.0; grad/=max(1,len(y))
                logits.backward(grad/micro)
            opt.step(); last=float(np.mean(losses))
            if step==1 or step%10==0 or step==steps:
                self._set(progress=min(95,20+int(75*step/max(1,steps))),message=f"학습 중... step {step:,}/{steps:,} · loss {last:.4f}",step=step,loss=last)
        tmp=CP.with_suffix(".learning.npz"); model.save(str(tmp)); os.replace(tmp,CP)
        meta["epoch"]=int(meta.get("epoch",0))+steps; meta["train_loss"]=last; meta["version"]="2.4.0-learning"
        MP.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        if self.reload_callback: self.reload_callback()
        return last

    def _run(self):
        try:
            lines=self.collect(); loss=self._train(lines,int(self.status["steps"]))
            if loss is None: self._set(running=False,phase="stopped",progress=100,message="학습을 중지했어.",finished_at=time.time())
            else: self._set(running=False,phase="complete",progress=100,message=f"학습 완료! loss={loss:.4f}",finished_at=time.time(),loss=loss)
        except Exception as exc:
            self._set(running=False,phase="error",progress=100,message="학습 중 오류가 발생했어.",finished_at=time.time(),error=repr(exc))
        finally:
            with self.lock: self.running=False
