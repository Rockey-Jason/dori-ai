import json, os, subprocess, sys, threading, time
from pathlib import Path

from .site_data import SiteData

ROOT = Path(__file__).resolve().parent.parent
CP = ROOT / "checkpoints" / "best.npz"
MP = Path(str(CP) + ".json")
STATUS = ROOT / "runtime" / "learning_status.json"
AUTO = ROOT / "data" / "learning" / "auto_corpus.txt"

class LearningManager:
    """Background bridge from the admin button to the large-data streaming trainer."""

    def __init__(self, reload_callback=None):
        self.reload_callback = reload_callback
        self.lock = threading.Lock()
        self.running = False
        self.stop_requested = False
        self.process = None
        self.status = {
            "running": False, "phase": "idle", "progress": 0,
            "message": "학습 대기 중", "examples": 0, "tokens": 0,
            "step": 0, "steps": 0, "loss": None, "started_at": None,
            "finished_at": None, "error": None
        }
        self._load()

    def _load(self):
        try:
            if STATUS.exists():
                self.status.update(json.loads(STATUS.read_text(encoding="utf-8")))
        except Exception:
            pass

    def _save(self):
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(self.status, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_status(self):
        with self.lock:
            return dict(self.status)

    def _set(self, **kw):
        with self.lock:
            self.status.update(kw)
            self._save()

    def _stop(self):
        with self.lock:
            return self.stop_requested

    def stop(self):
        with self.lock:
            if not self.running:
                return False
            self.stop_requested = True
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                except Exception:
                    pass
            self.status["message"] = "학습 중지 요청을 처리하는 중..."
            self._save()
            return True

    def start(self, epochs=None):
        with self.lock:
            if self.running:
                return False, "이미 학습 중이야."
            n = max(1, min(1000, int(epochs or os.getenv("DORI_TRAIN_EPOCHS", "10"))))
            self.running = True
            self.stop_requested = False
            self.process = None
            self.status.update({
                "running": True, "phase": "starting", "progress": 0,
                "message": "대용량 학습을 준비하는 중...", "examples": 0,
                "tokens": 0, "step": 0, "steps": n, "epochs": n,
                "loss": None, "started_at": time.time(),
                "finished_at": None, "error": None
            })
            self._save()
            threading.Thread(target=self._run, args=(n,), daemon=True, name="dori-streaming-learning").start()
            return True, "대용량 학습을 시작했어."

    def collect(self):
        """Export live Dori-site knowledge into the streaming data directory."""
        rows = []
        site = SiteData()

        try:
            for row in site.public_news_all():
                n = row.get("news_number")
                news = row.get("rockey_news", "")
                if news:
                    rows.append(f"돌이신문 제{n}호: {news}")
                if row.get("question"):
                    rows.append(f"돌이신문 퀴즈: {row.get('question','')}")
        except Exception:
            pass

        try:
            for row in site.stock():
                rows.append(
                    "돌돌증권 종목: "
                    f"{row.get('name','')} {row.get('ticker','')} "
                    f"설명: {row.get('description','')} "
                    f"특징: {row.get('characteristics','')} "
                    f"위험도: {row.get('risk_label','')}"
                )
        except Exception:
            pass

        AUTO.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        clean = []
        for row in rows:
            row = " ".join(str(row).split()).strip()
            if len(row) >= 20 and row.casefold() not in seen:
                seen.add(row.casefold())
                clean.append(row)
        AUTO.write_text("\n".join(clean) + ("\n" if clean else ""), encoding="utf-8")
        self._set(
            phase="collected",
            progress=8,
            message=f"사이트 학습 자료 {len(clean):,}개를 준비했어.",
            examples=len(clean)
        )
        return clean

    def _run(self, epochs):
        try:
            self.collect()
            if self._stop():
                self._set(running=False, phase="stopped", progress=100, message="학습을 중지했어.", finished_at=time.time())
                return

            cmd = [
                sys.executable, str(ROOT / "train_final.py"),
                "--epochs", str(epochs),
                "--seq-len", os.getenv("DORI_TRAIN_SEQ_LEN", "128"),
                "--batch-size", os.getenv("DORI_TRAIN_BATCH_SIZE", "8"),
                "--dim", os.getenv("DORI_TRAIN_DIM", "64"),
                "--heads", os.getenv("DORI_TRAIN_HEADS", "4"),
                "--layers", os.getenv("DORI_TRAIN_LAYERS", "3"),
                "--ff-dim", os.getenv("DORI_TRAIN_FF_DIM", "256"),
                "--lr", os.getenv("DORI_TRAIN_LR", "2e-4"),
                "--grad-clip", os.getenv("DORI_TRAIN_GRAD_CLIP", "1.0")
            ]

            self._set(
                phase="training",
                progress=10,
                message=f"대용량 Transformer 학습 중... {epochs} epoch"
            )

            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            last_line = ""
            while True:
                if self._stop() and self.process.poll() is None:
                    try:
                        self.process.terminate()
                    except Exception:
                        pass

                line = self.process.stdout.readline() if self.process.stdout else ""
                if line:
                    last_line = line.strip()
                    self._parse_line(last_line, epochs)
                elif self.process.poll() is not None:
                    break
                else:
                    time.sleep(0.1)

            code = self.process.wait()
            self.process = None

            if self._stop():
                self._set(
                    running=False, phase="stopped", progress=100,
                    message="학습을 중지했어.", finished_at=time.time()
                )
                return

            if code != 0:
                raise RuntimeError(last_line or f"train_final.py exited with code {code}")

            if self.reload_callback:
                self.reload_callback()

            best_meta = {}
            best_meta_path = Path(str(CP) + ".json")
            if best_meta_path.exists():
                try:
                    best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            self._set(
                running=False,
                phase="complete",
                progress=100,
                message=f"학습 완료! best val loss={best_meta.get('val_loss', '—')}",
                loss=best_meta.get("val_loss"),
                finished_at=time.time()
            )

        except Exception as exc:
            self.process = None
            self._set(
                running=False, phase="error", progress=100,
                message="학습 중 오류가 발생했어.", finished_at=time.time(),
                error=repr(exc)
            )
        finally:
            with self.lock:
                self.running = False
                self.process = None

    def _parse_line(self, line, epochs):
        import re
        m = re.search(r"Epoch\s+(\d+).*train\s+([0-9.]+).*val\s+([0-9.]+)", line)
        if m:
            epoch = int(m.group(1))
            train_loss = float(m.group(2))
            val_loss = float(m.group(3))
            self._set(
                phase="training",
                progress=min(95, 10 + int(85 * epoch / max(1, epochs))),
                message=f"학습 중... epoch {epoch:,}/{epochs:,} · train {train_loss:.4f} · val {val_loss:.4f}",
                step=epoch,
                steps=epochs,
                loss=val_loss
            )
        elif line:
            self._set(message=line[-500:])
