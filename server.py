"""Render entry point for Dori AI.
Keeps the existing dori_server implementation and only adapts the bind address
for hosted environments.
"""
import os

os.environ.setdefault("DORI_HOST", "0.0.0.0")

from dori_server import Handler, ThreadingHTTPServer

host = os.getenv("DORI_HOST", "0.0.0.0")
port = int(os.getenv("PORT", os.getenv("DORI_PORT", "8000")))

print(f"Dori AI server listening on {host}:{port}", flush=True)

ThreadingHTTPServer((host, port), Handler).serve_forever()
