"""Probe server for Runpod load-balancer endpoints.

Reports the HOST NVIDIA driver (from /proc/driver/nvidia/version, which is
host procfs visible inside the container) and nvidia-smi output. This is the
number that decides whether a torch cu128/cu13 build can run on the host.
Health endpoint: GET /ping -> 200 pong (Runpod LB health check).
Everything else: plain-text driver report.
"""
import http.server
import os
import subprocess
import urllib.request

PORT = int(os.environ.get("PORT", "8000"))


def report_bytes():
    out = b""
    try:
        out += subprocess.check_output(["nvidia-smi"], timeout=10) + b"\n"
    except Exception as e:
        out += ("NVIDIA_SMI_UNAVAILABLE: %r\n" % (e,)).encode()
    try:
        out += b"PROC_DRIVER:\n" + open("/proc/driver/nvidia/version", "rb").read()
    except Exception as e:
        out += ("PROC_UNAVAILABLE: %r\n" % (e,)).encode()
    return out


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, out):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.startswith("/ping"):
            self._send(200, b"pong")
            return
        self._send(200, report_bytes())

    def log_message(self, *a):
        pass


class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print("probe server listening on port", PORT, flush=True)
    Server(("0.0.0.0", PORT), H).serve_forever()
