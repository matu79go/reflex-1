"""Reflex-1 HTTP endpoint.

POST /v1/decisions   Jev-compatible request shape, plus Reflex-1 extensions (mode, skill, steps, trace, image, type=text)
GET  /health

Usage:
  python -m reflex.server --skills skills.json --port 8097 --quant nf4
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import Reflex, load_skills


def _image(v):
    """image: base64 string (optionally a data URL) or a local file path."""
    from PIL import Image

    if v is None:
        return None
    if v.startswith("data:"):
        v = v.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(v, validate=True))).convert("RGB")
    except Exception:  # noqa: BLE001 not base64: treat as a path
        return Image.open(v).convert("RGB")


def decide(rf: Reflex, req: dict, lock: threading.Lock) -> dict:
    state = req["state"] if isinstance(req.get("state"), str) else json.dumps(req.get("state", ""), ensure_ascii=False)
    mode = req.get("mode", "instant")
    img = _image(req.get("image"))
    out = {}
    with lock:
        for name, q in req["questions"].items():
            if q.get("type") == "text":
                out[name] = {"text": rf.write(state, q["instructions"], int(q.get("max_tokens", 256)))}
            elif mode == "deep":
                skill = req.get("skill")
                if skill not in rf.skills:
                    raise ValueError(f"unknown skill: {skill} (available: {list(rf.skills)})")
                out[name] = rf.deep(state, skill, req.get("steps"), bool(req.get("trace")))
            else:
                out[name] = rf.instant(state, q["instructions"], q["criteria"], img)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--skills", default="", help="skills.json (deep mode); empty = instant only")
    ap.add_argument("--quant", default="nf4", choices=["none", "int8", "nf4"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8097)
    a = ap.parse_args()
    rf = Reflex(a.base, load_skills(a.skills) if a.skills else None, a.quant)
    lock = threading.Lock()
    print(f"READY skills={list(rf.skills)} quant={a.quant} port={a.port}", flush=True)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "skills": list(rf.skills)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/decisions":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                t0 = time.perf_counter()
                ans = decide(rf, req, lock)
                self._send(200, {"answers": ans, "mode": req.get("mode", "instant"), "latency_ms": round((time.perf_counter() - t0) * 1000)})
            except Exception as e:  # noqa: BLE001 bad requests get the reason back
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
