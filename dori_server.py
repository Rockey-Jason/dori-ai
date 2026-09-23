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
\n
site = SiteData()

def reload_bot():
    global meta, tok, model, bot
    meta = json.loads(MP.read_text(encoding="utf-8"))
    tok = BPETokenizer.load(ROOT / meta["tokenizer"])
    model = load_model(str(CP), meta["model_config"])
    bot = ResponseEngine(tok, model)

learner = LearningManager(reload_callback=reload_bot)

# CORS is intentionally open because the Dori site is hosted separately from Render.
# Authorization is accepted so the frontend can pass the Supabase access token.
CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Accept, Authorization",
    "Access-Control-Max-Age": "86400",
}


def _knowledge_size():
    """Return knowledge count without assuming LocalKnowledge has a size() method."""
    try:
        value = getattr(bot.kb, "size", None)
        if callable(value):
            return int(value())
        if value is not None:
            return int(value)
        for name in ("entries", "data", "items", "records"):
            value = getattr(bot.kb, name, None)
            if value is not None:
                return len(value)
    except Exception:
        pass
    return None


def _supabase_user_from_token(token):
    """Validate a Supabase access token and return the authenticated user id.

    Authentication is optional: if Supabase server variables are not configured,
    anonymous chat still works. If a token is supplied and validation is possible,
    the server never trusts a client-provided user_id over the verified token.
    """
    if not token:
        return None

    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_key = os.getenv("SUPABASE_ANON_KEY", "")
    if not supabase_url or not supabase_key:
        # Do not pretend that a user_id supplied by the browser is authenticated.
        return None

    req = urllib.request.Request(
        supabase_url + "/auth/v1/user",
        headers={
            "apikey": supabase_key,
            "Authorization": "Bearer " + token,
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
            uid = str(payload.get("id", "")).strip()
            return uid or None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _sse_event(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def cors(self):
        for k, v in CORS.items():
            self.send_header(k, v)

    def send(self, code, obj, typ="application/json; charset=utf-8"):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", typ)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/training/status":
            token = self.headers.get("Authorization", "")
            token = token[7:].strip() if token.lower().startswith("bearer ") else ""
            uid = _supabase_user_from_token(token)
            if not uid or not site.is_admin(uid):
                self.send(403, {"error": "관리자만 학습 모드를 사용할 수 있어."})
                return
            self.send(200, learner.get_status())
            return
        if path == "/health":
            self.send(
                200,
                {
                    "status": "ok",
                    "service": "dori-ai",
                    "version": meta.get("version"),
                    "knowledge_entries": _knowledge_size(),
                    "web_search": bot.web_enabled,
                    "model": meta.get("model_config"),
                },
            )
            return
        self.send(200, {"service": "Dori AI", "status": "online"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/training/start":
            auth_header = self.headers.get("Authorization", "")
            token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
            uid = _supabase_user_from_token(token)
            if not uid or not site.is_admin(uid):
                self.send(403, {"error": "관리자만 학습 모드를 사용할 수 있어."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                steps = int(data.get("steps") or os.getenv("DORI_TRAIN_STEPS", "600"))
                steps = max(50, min(5000, steps))
                ok, message = learner.start(steps)
                self.send(202 if ok else 409, {"ok": ok, "message": message, "status": learner.get_status()})
            except Exception as exc:
                self.send(400, {"error": str(exc)})
            return
        if path == "/training/stop":
            auth_header = self.headers.get("Authorization", "")
            token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
            uid = _supabase_user_from_token(token)
            if not uid or not site.is_admin(uid):
                self.send(403, {"error": "관리자만 학습 모드를 사용할 수 있어."})
                return
            self.send(200, {"ok": learner.stop(), "status": learner.get_status()})
            return
        if path != "/chat":
            self.send(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16384:
                raise ValueError("request too large or empty")

            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            text = str(data.get("message", data.get("text", ""))).strip()
            mode = str(data.get("mode", "fast"))
            stream = bool(data.get("stream", False))

            if not text or len(text) > 2000:
                raise ValueError("text must be 1-2000 characters")

            # Prefer the verified Supabase token. A browser-supplied user_id is only
            # treated as an anonymous routing hint and is never called authenticated.
            auth_header = self.headers.get("Authorization", "")
            token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
            verified_uid = _supabase_user_from_token(token)
            client_uid = str(data.get("user_id", "")).strip() or None
            uid = verified_uid or client_uid
            authenticated = verified_uid is not None

            print(
                f"Dori AI request mode={mode!r} authenticated={authenticated} user_id={uid!r}",
                flush=True,
            )

            answer = bot.reply(text, user_id=uid)
            answer = "" if answer is None else str(answer).strip()

            if not answer:
                raise RuntimeError("Dori AI generated an empty response")

            if stream:
                # The current inference engine returns a completed answer. We expose
                # it through the same SSE contract as the frontend, in small chunks,
                # so the UI can render progressively without changing the model.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-transform")
                self.send_header("Connection", "keep-alive")
                self.cors()
                self.end_headers()
                try:
                    chunk_size = 24
                    for i in range(0, len(answer), chunk_size):
                        self.wfile.write(
                            _sse_event({"type": "token", "content": answer[i : i + chunk_size]}).encode("utf-8")
                        )
                        self.wfile.flush()
                    self.wfile.write(_sse_event({"type": "done"}).encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            self.send(
                200,
                {
                    "answer": answer,
                    "mode": mode,
                    "authenticated": authenticated,
                    "user_id": verified_uid,
                },
            )

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
