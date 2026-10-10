import json
import os
import urllib.error
import urllib.request
import gc
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Cap request-handler threads as well as chat inference.
class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 32

    def __init__(self, server_address, request_handler, max_workers=8):
        self._worker_slots = threading.BoundedSemaphore(max(2, min(16, int(max_workers))))
        super().__init__(server_address, request_handler)

    def process_request(self, request, client_address):
        self._worker_slots.acquire()
        try:
            super().process_request(request, client_address)
        except Exception:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_slots.release()



from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from dori_ai.response_engine import ResponseEngine
from dori_ai.llm_provider import model_name as llm_model_name
from dori_ai.site_data import SiteData
from dori_ai.learning import LearningManager
from dori_ai import world_knowledge

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

# The Render Free instance has 512 MB RAM. ThreadingHTTPServer can otherwise
# run several Transformer inference graphs at once and multiply peak memory.
CHAT_CONCURRENCY = max(1, int(os.getenv("DORI_CHAT_CONCURRENCY", "1")))
CHAT_SEMAPHORE = threading.BoundedSemaphore(CHAT_CONCURRENCY)

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

def _verified_admin(handler):
    token = _token_from_request(handler)
    uid = _supabase_user_from_token(token)
    if not uid:
        print("Dori AI admin auth: invalid/missing Supabase access token", flush=True)
        return None

    # First try the verified user's token. This respects normal RLS.
    authorized_site = site.with_token(token)
    if authorized_site.is_admin(uid):
        print(f"Dori AI admin auth: allowed user={uid}", flush=True)
        return uid

    # Server-side fallback for projects whose users table policy blocks
    # authenticated REST reads. Never trust a browser-supplied admin flag.
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if service_key:
        # Service-role key must be used for BOTH API-key and bearer headers.
        service_site = SiteData(access_token=service_key, api_key=service_key)
        if service_site.is_admin(uid):
            print(f"Dori AI admin auth: allowed via service role user={uid}", flush=True)
            return uid

    print(f"Dori AI admin auth: verified user but not admin user={uid}", flush=True)
    return None

def _sse_event(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

def _deterministic_fact(text):
    # Keep unambiguous, stable facts out of probabilistic generation.
    q="".join(str(text).lower().split())
    if q in {
        "대한민국의수도는어디야?",
        "대한민국의수도는어디인가?",
        "대한민국의수도는어디인가요?",
        "대한민국의수도는?",
        "대한민국의수도는서울이야?",
        "서울은대한민국의수도야?",
        "서울이대한민국의수도야?",
    }:
        return "대한민국의 수도는 서울이야. 🇰🇷"
    return None

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def cors(self):
        for k, v in CORS.items():
            self.send_header(k, v)

    def end_headers(self):
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
        self.close_connection = True

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self.send(200, {
                "status": "ok",
                "service": "dori-ai",
                "version": "3.1.0-world-knowledge",
                "knowledge_entries": _knowledge_size(),
                "web_search": bot.web_enabled,
                "chat_provider": {"enabled": bot.llm_enabled, "model": llm_model_name()},
                "world_knowledge": {"enabled": True, "source": "Wikipedia API", "cache_ttl_seconds": world_knowledge._TTL},
                "model": meta.get("model_config"),
                "training": learner.get_status(),
            })
            return
        if path == "/training/status":
            if not _verified_admin(self):
                self.send(403, {"error": "관리자만 학습 모드를 사용할 수 있어."})
                return
            self.send(200, learner.get_status())
            return
        self.send(200, {"service": "Dori AI", "status": "online"})

    def _admin(self):
        return _verified_admin(self)

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
                    mode = str(data.get("mode") or "normal")
                    epochs = data.get("epochs")
                    max_batches = data.get("max_batches")
                    batch_size = data.get("batch_size")
                    ok, message = learner.start(epochs, mode=mode, max_batches=max_batches, batch_size=batch_size)
                    self.send(202 if ok else 409, {"ok": ok, "message": message, "status": learner.get_status()})
                else:
                    self.send(200, {"ok": learner.stop(), "status": learner.get_status()})
            except Exception as exc:
                self.send(400, {"error": str(exc)})
            return

        if path != "/chat":
            self.send(404, {"error": "not found"})
            return

        acquired_chat = False
        try:
            if path == "/chat":
                acquired_chat = CHAT_SEMAPHORE.acquire(timeout=20)
                if not acquired_chat:
                    self.send(503, {"error": "Dori AI가 현재 다른 요청을 처리하고 있어. 잠시 후 다시 시도해줘.", "type": "busy"})
                    return
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
            uid = verified_uid
            print(f"Dori AI request mode={mode!r} authenticated={bool(uid)} user_id={uid!r}", flush=True)

            answer = _deterministic_fact(text)
            if answer is None:
                answer = str(bot.reply(
                    text,
                    user_id=uid,
                    access_token=_token_from_request(self),
                    mode=mode
                ) or "").strip()
            else:
                print("Dori AI deterministic fact route", flush=True)
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
                finally:
                    self.close_connection = True
                return

            self.send(200, {"answer": answer, "mode": mode, "authenticated": bool(uid), "user_id": uid})
        except ValueError as exc:
            self.send(400, {"error": str(exc), "type": "request_error"})
        except Exception as exc:
            print("Dori AI /chat error:", repr(exc), flush=True)
            self.send(500, {"error": "Dori AI 내부 오류가 발생했어.", "type": "server_error"})
        finally:
            if acquired_chat:
                CHAT_SEMAPHORE.release()
                # Release temporary NumPy/autograd objects before the next
                # request. This is a safety valve, not a substitute for limits.
                gc.collect()

    def log_message(self, *args):
        pass

if __name__ == "__main__":
    host = os.getenv("DORI_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("DORI_PORT", "8000")))
    print(f"Dori AI server listening on {host}:{port}", flush=True)
    max_http_workers = int(os.getenv("DORI_HTTP_MAX_WORKERS", "8"))
    BoundedThreadingHTTPServer((host, port), H, max_workers=max_http_workers).serve_forever()
