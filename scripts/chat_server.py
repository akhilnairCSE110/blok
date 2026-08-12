#!/usr/bin/env python3
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import blok.runtime as blk

ROOT = Path(__file__).parents[1]
WEB = ROOT / "web"
RUN_LOCK = threading.Lock()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, model: str, max_tokens: int, max_time: float):
        super().__init__(address, Handler)
        self.model, self.max_tokens, self.max_time = model, max_tokens, max_time


class Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        if self.path == "/":
            self.file(WEB / "index.html", "text/html; charset=utf-8")
        elif self.path == "/app.js":
            self.file(WEB / "app.ts", "text/javascript; charset=utf-8")
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.json_response(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1_000_000:
                raise ValueError("request body must be between 1 byte and 1 MB")
            body = json.loads(self.rfile.read(size))
            prompt, tokens = body.get("prompt"), body.get("max_tokens", 1000)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be non-empty text")
            if type(tokens) is not int or not 0 < tokens <= self.server.max_tokens:
                raise ValueError(f"max_tokens must be between 1 and {self.server.max_tokens}")
            if not RUN_LOCK.acquire(blocking=False):
                self.json_response(409, {"error": "another generation is already running"})
                return
            try:
                result = blk.generate_report(model_dir=self.server.model, prompt=prompt, max_tokens=tokens,
                                             max_time=self.server.max_time)
            finally:
                RUN_LOCK.release()
            self.json_response(200, {"text": result.text, "input_tokens": result.input_tokens,
                                     "output_tokens": result.output_tokens, "finish_reason": result.finish_reason})
        except (ValueError, json.JSONDecodeError, blk.BlokRuntimeError) as error:
            self.json_response(400, {"error": str(error)})

    def file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.security_headers()
        self.end_headers()
        self.wfile.write(data)

    def json_response(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.security_headers()
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass

    def security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; connect-src 'self'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local Blok chat UI")
    parser.add_argument("--model", default=os.getenv("BLOK_MODEL"), help="runtime index or metadata directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-output-tokens", type=int, default=10_000)
    parser.add_argument("--max-time", type=float, default=float(os.getenv("BLOK_CHAT_MAX_TIME", "2592000")))
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("set BLOK_MODEL or pass --model")
    if not 0 < args.port < 65536 or args.max_output_tokens <= 0 or args.max_time <= 0:
        raise SystemExit("port and generation limits must be positive")
    server = Server((args.host, args.port), args.model, args.max_output_tokens, args.max_time)
    print(f"Blok chat: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
