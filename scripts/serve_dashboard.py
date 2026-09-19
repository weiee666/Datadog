"""Start the DDOG dashboard with one configured local URL.

The port lives in dashboard_config.json so the project has a single source of
truth instead of ad-hoc 8899 / 8901 / 8910 services.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socket
import socketserver
import subprocess
import sys
import webbrowser
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "dashboard_config.json"


def load_config() -> dict:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    return {
        "host": "127.0.0.1",
        "port": 8910,
        "site_dir": "site",
        "auto_rebuild_payload": True,
        "open_browser": False,
    }


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) != 0


def rebuild_payload() -> None:
    script = ROOT / "scripts" / "build_dashboard.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Serve the DDOG dashboard.")
    parser.add_argument("--port", type=int, default=int(cfg.get("port", 8910)))
    parser.add_argument("--host", default=str(cfg.get("host", "127.0.0.1")))
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser.")
    args = parser.parse_args()

    site_dir = ROOT / str(cfg.get("site_dir", "site"))
    if not site_dir.exists():
        raise SystemExit(f"site directory not found: {site_dir}")

    if cfg.get("auto_rebuild_payload", True) and not args.no_rebuild:
        rebuild_payload()

    if not port_available(args.host, args.port):
        raise SystemExit(
            f"Port {args.port} is already in use. Close the old dashboard service, "
            f"or edit {CONFIG.name} / pass --port."
        )

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(site_dir))
    socketserver.TCPServer.allow_reuse_address = True
    url = f"http://{args.host}:{args.port}/"

    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"DDOG dashboard: {url}")
        print(f"Config: {CONFIG}")
        print("Press Ctrl+C to stop.")
        if args.open or cfg.get("open_browser", False):
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
