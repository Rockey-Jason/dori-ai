import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from dori_ai.response_engine import ResponseEngine
from dori_ai.site_data import SiteData
from dori_ai.learning import LearningManager

ROOT = Path(__file__).resolve().parent
CP = ROOT / "checkpoints" / "best.npz"
MP = Path(str(CP) + ".json")

if not CP.exists() or not MP.exists():
    raise SystemExit("Missing checkpoints/best.npz or metadata.")

meta = json.loads(MP.read_text(encoding="utf-8"))
tok = BPETokenizer.load(ROOT / meta["tokenizer"])
model = load_model(str(CP), meta["model_config"])
bot = ResponseEngine(tok, model)
site = SiteData()

def reload_bot():
    global meta, tok, model, bot
    meta = json.loads(MP.read_text(encoding="utf-8"))
    tok = BPETokenizer.load(ROOT / meta["tokenizer"])
    model = load_model(str(CP), meta["model_config"])
    bot = ResponseEngine(tok, model)

learner = LearningManager(reload_callback=reload_bot)

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Accept, Authorization",
    "Access-Control-Max-Age": "86400",
}

def _knowledge_size():
    try:
        value = getattr(bot.kb, "size", None)
        return int(value()) if callable(value) else int(value or 0)
    except Exception:
        return None

def _supabase_user_from_token(token):
    if not token:
        return None
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return None
    req = urllib.request.Request(
        url + "/auth/v1/user",
        headers={"apikey": key, "Authorization": "Bearer " + token, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
            return str(payload.get("id", "")).strip() or None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None

def _token_from_request(handler):
    h = handler.headers.get("Authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else ""

def _sse_event(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def cors(self):
        for k, v in CORS.items():
            self.send_header(k, v)

    def end_headers(self):
        # Render/proxy environments can return errors or intermediate responses
        # where CORS headers are otherwise easy to miss. Attach CORS at the
        # final header-flush point so every response, including 204/404/500
        # and streaming responses, gets the same policy.
        self.cors()
        super().end_headers()

    def send(self, code, obj, typ="application/json; charset=utf-8"):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", typ)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self.send(200, {
                "status": "ok",
                "service": "dori-ai",
                "version": "2.6.1-corsfix",
                "knowledge_entries": _knowledge_size(),
                "web_search": bot.web_enabled,
                "model": meta.get("model_config"),
                "training": learner.get_status(),
            })
            return
        if path == "/training/status":
            uid = _supabase_user_from_token(_token_from_request(self))
            if not uid or not site.is_admin(uid):
                self.send(403, {"error": "관리자만 학습 모드를 사용할 수 있어."})
                return
            self.send(200, learner.get_status())
            return
        self.send(200, {"service": "Dori AI", "status": "online"})

    def _admin(self):
        uid = _supabase_user_from_token(_token_from_request(self))
        return uid if uid and site.is_admin(uid) else None

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path in ("/training/start", "/training/stop", "/knowledge/reload"):
            if not self._admin():
                self.send(403, {"error": "관리자만 학습 모드를 사용할 수 있어."})
                return
            try:
                if path == "/knowledge/reload":
                    reload_bot()
                    self.send(200, {"ok": True, "knowledge_entries": _knowledge_size()})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                if path.endswith("/start"):
                    steps = max(50, min(5000, int(data.get("steps") or os.getenv("DORI_TRAIN_STEPS", "600"))))
                    ok, message = learner.start(steps)
                    self.send(202 if ok else 409, {"ok": ok, "message": message, "status": learner.get_status()})
                else:
                    self.send(200, {"ok": learner.stop(), "status": learner.get_status()})
            except Exception as exc:
                self.send(400, {"error": str(exc)})
            return

        if path != "/chat":
            self.send(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16384:
                raise ValueError("request too large or empty")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(data.get("message", data.get("text", ""))).strip()
            mode = str(data.get("mode", "fast"))
            stream = bool(data.get("stream", False))
            if not text or len(text) > 2000:
                raise ValueError("text must be 1-2000 characters")

            verified_uid = _supabase_user_from_token(_token_from_request(self))
            # Never trust a browser-supplied user_id for permission checks.
            uid = verified_uid
            print(f"Dori AI request mode={mode!r} authenticated={bool(uid)} user_id={uid!r}", flush=True)

            answer = str(bot.reply(text, user_id=uid, access_token=_token_from_request(self), mode=mode) or "").strip()
            if not answer:
                raise RuntimeError("Dori AI generated an empty response")

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-transform")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    for i in range(0, len(answer), 24):
                        self.wfile.write(_sse_event({"type": "token", "content": answer[i:i+24]}).encode("utf-8"))
                        self.wfile.flush()
                    self.wfile.write(_sse_event({"type": "done"}).encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            self.send(200, {"answer": answer, "mode": mode, "authenticated": bool(uid), "user_id": uid})
        except ValueError as exc:
            self.send(400, {"error": str(exc), "type": "request_error"})
        except Exception as exc:
            print("Dori AI /chat error:", repr(exc), flush=True)
            self.send(500, {"error": "Dori AI 내부 오류가 발생했어.", "type": "server_error"})

    def log_message(self, *args):
        pass

if __name__ == "__main__":
    host = os.getenv("DORI_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("DORI_PORT", "8000")))
    print(f"Dori AI server listening on {host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()
