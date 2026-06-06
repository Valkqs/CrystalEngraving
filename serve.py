"""Zero-dependency static file server (stdlib only)."""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 5000
ROOT = Path(__file__).parent / "static"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print(f"Serving at {url}")
        print("Press Ctrl+C to stop.")
        webbrowser.open(url)
        httpd.serve_forever()
