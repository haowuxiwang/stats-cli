"""REST API server for stats-cli-py.

Usage:
    python serve.py --port 8080
    python -m stats_engine.serve --port 8080

Endpoints:
    POST /api/v1/analyze     -> JSON in, JSON out (same as handler())
    POST /api/v1/discover    -> list commands
    GET  /api/v1/health      -> health check
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from main import handler


class StatsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/v1/health":
            self._respond(200, {"status": "ok", "service": "stats-cli-py"})
        else:
            self._respond(404, {"status": "error", "message": "Not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            self._respond(400, {"status": "error", "message": f"Invalid JSON: {e}"})
            return

        if self.path == "/api/v1/discover":
            result = handler({"command": "discover", "params": data.get("params", {})})
            self._respond(200, result)
        elif self.path == "/api/v1/analyze":
            result = handler(data)
            status = 200 if result.get("status") != "error" else 400
            self._respond(status, result)
        else:
            self._respond(404, {"status": "error", "message": "Not found"})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    import argparse

    parser = argparse.ArgumentParser(description="stats-cli-py REST API server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), StatsHandler)
    print(f"stats-cli-py API server running on http://{args.host}:{args.port}")
    print("  POST /api/v1/analyze  - Run analysis")
    print("  POST /api/v1/discover - List commands")
    print("  GET  /api/v1/health   - Health check")
    server.serve_forever()


if __name__ == "__main__":
    main()
