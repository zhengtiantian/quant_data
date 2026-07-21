#!/usr/bin/env python3
"""
Local SLM reverse proxy: limits concurrent forwarding to LM Studio to avoid multiple workers overwhelming the upstream.
Usage: python slm_proxy.py [--port 11435] [--upstream http://192.168.31.226:1234] [--concurrency 2]
"""
import argparse
import threading
import urllib.request
import urllib.parse
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=11435)
parser.add_argument("--upstream", default="http://192.168.31.226:1234")
parser.add_argument("--concurrency", type=int, default=2)
args = parser.parse_args()

UPSTREAM = args.upstream.rstrip("/")
_sem = threading.Semaphore(args.concurrency)
_lock = threading.Lock()
_stats = {"total": 0, "active": 0, "errors": 0}

class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass  # suppress access log

    def _forward(self, body: bytes | None):
        url = UPSTREAM + self.path
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "transfer-encoding")}
        if body:
            headers["Content-Length"] = str(len(body))

        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        with _lock:
            _stats["total"] += 1
            _stats["active"] += 1
        with _sem:
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() not in ("transfer-encoding",):
                            self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(data)
            except Exception as e:
                with _lock:
                    _stats["errors"] += 1
                self.send_error(502, str(e))
        with _lock:
            _stats["active"] -= 1

    def do_GET(self):
        self._forward(None)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._forward(body)

    def do_OPTIONS(self):
        self._forward(None)


print(f"🚀 SLM Proxy: localhost:{args.port} → {UPSTREAM}  (concurrency={args.concurrency})")
ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler).serve_forever()
