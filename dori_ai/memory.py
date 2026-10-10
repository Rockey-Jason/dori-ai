import json, math, re, time, os
from pathlib import Path

_MAX_ITEMS = max(50, min(1000, int(os.getenv("DORI_MEMORY_MAX_ITEMS", "300"))))
_MAX_ITEM_CHARS = 2000
_MAX_FILE_BYTES = 1_000_000


class MemoryStore:
    def __init__(self, path="memory/dori_memory.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items = []
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            if self.path.stat().st_size > _MAX_FILE_BYTES:
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return
            clean = []
            for item in data[-_MAX_ITEMS:]:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()[:_MAX_ITEM_CHARS]
                if text:
                    clean.append({
                        "text": text,
                        "kind": str(item.get("kind", "fact"))[:40],
                        "time": float(item.get("time", 0) or 0),
                    })
            self.items = clean
        except (OSError, ValueError, TypeError):
            self.items = []

    def save(self):
        self.items = self.items[-_MAX_ITEMS:]
        payload = json.dumps(self.items, ensure_ascii=False, separators=(",", ":"))
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)

    def _terms(self, text):
        return set(re.findall(r"[\w가-힣]+", str(text).lower()))

    def add(self, text, kind="fact"):
        text = str(text or "").strip()[:_MAX_ITEM_CHARS]
        if not text:
            return
        self.items.append({"text": text, "kind": str(kind)[:40], "time": time.time()})
        self.items = self.items[-_MAX_ITEMS:]
        self.save()

    def search(self, query, k=5):
        query = str(query or "")
        qt = self._terms(query)
        if not qt:
            return []
        scored = []
        for item in self.items:
            if not isinstance(item, dict):
                continue
            item_text = str(item.get("text", ""))
            st = self._terms(item_text)
            overlap = len(qt & st)
            if query.strip() and query.strip().casefold() in item_text.casefold():
                overlap += 2
            score = overlap / (math.sqrt(len(qt) * len(st)) + 1e-9)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:max(0, min(20, int(k)))]]

    def context(self, query, k=5):
        return "\n".join("- " + item["text"] for item in self.search(query, k))
