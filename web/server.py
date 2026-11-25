"""Serve the dashboard and REST endpoints for one local HZL experiment."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import subprocess
from threading import Lock

ROOT = Path(__file__).resolve().parent
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
}


class Dashboard(BaseHTTPRequestHandler):
    timeout = 10

    def reply(self, status, payload, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def error(self, status, message):
        self.reply(status, json.dumps({"error": message}).encode())

    def do_GET(self):
        if self.path in ASSETS:
            filename, content_type = ASSETS[self.path]
            self.reply(200, (ROOT / filename).read_bytes(), content_type)
        elif self.path == "/api/state":
            self.reply(200, self.server.state.encode())
        else:
            self.error(404, "Unknown endpoint")

    def do_POST(self):
        routes = {"/api/preview": "preview", "/api/reset": "reset", "/api/save": "save"}
        if self.path not in (*routes, "/api/step", "/api/paint"):
            self.error(404, "Unknown endpoint")
            return
        if self.headers.get_content_type() != "application/json":
            self.error(415, "Expected application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            if self.path == "/api/step":
                automatic = body.get("automatic")
                if type(automatic) is not bool:
                    raise ValueError("Expected a boolean automatic setting")
                command = "auto" if automatic else "manual"
            elif self.path == "/api/paint":
                field, x, y, value = (body.get(key) for key in ("field", "x", "y", "value"))
                if (field not in ("temperature", "target") or type(x) is not int or type(y) is not int
                        or type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError("Invalid brush coordinates or value")
                command = f"{field} {x} {y} {value}"
            else:
                command = routes[self.path]
        except (ValueError, OverflowError) as error:
            self.error(400, str(error))
            return
        try:
            with self.server.engine_lock:
                engine = self.server.engine
                engine.stdin.write(command + "\n")
                engine.stdin.flush()
                result = engine.stdout.readline()
                if not result:
                    raise RuntimeError("Simulation process exited")
                if not result.startswith("!"):
                    self.server.state = result
        except (OSError, RuntimeError) as error:
            self.error(503, str(error))
            return
        if result.startswith("!"):
            self.error(422, result[1:].strip())
        else:
            self.reply(200, result.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--engine", default="build/cuda/hzl-web-engine")
    parser.add_argument("settings", nargs=argparse.REMAINDER, help="Experiment options after --")
    args = parser.parse_args()
    settings = args.settings[1:] if args.settings[:1] == ["--"] else args.settings
    engine = subprocess.Popen([str(Path(args.engine).resolve()), *settings],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        initial = engine.stdout.readline()
        if not initial:
            raise SystemExit("Could not initialize the simulation.")
        with ThreadingHTTPServer((args.host, args.port), Dashboard) as server:
            server.engine = engine
            server.engine_lock = Lock()
            server.state = initial
            print(f"HZL dashboard listening on http://{args.host}:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stdin.close()
        engine.wait()


if __name__ == "__main__":
    main()
