"""Test page: upload a recording, watch the rough draft appear window by window, then the final
transcript (the same pipeline as transcribe.py and the phone apps). Any format ffmpeg reads works
(wav, mp3, m4a, aac, flac, ogg, opus, webm, mp4/mov…).

    python webtest/server.py   # then open http://127.0.0.1:8770
"""
import json, os, subprocess, sys, tempfile, time, warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from transcribe import Transcriber  # noqa: E402

PORT = int(os.environ.get("PORT", "8770"))
print("loading model…", flush=True)
TR = Transcriber(os.environ.get("MODELS", os.path.join(ROOT, "models")), os.environ.get("BACKEND", "auto"))
LANGS = TR.langs
print("ready", flush=True)


def decode_audio(data: bytes) -> np.ndarray:
    """Any audio/video file -> 16 kHz mono float32 (ffmpeg). Through a file, not a pipe: an
    m4a/mov often has its index at the end, which a pipe can't seek to."""
    with tempfile.NamedTemporaryFile() as f:
        f.write(data); f.flush()
        p = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", f.name,
                            "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise ValueError(p.stderr.decode(errors="ignore").strip() or "ffmpeg could not read the file")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def transcribe(data, langs, simplified, emit):
    t0 = time.time()
    emit({"stage": "decode"})
    wav = decode_audio(data)
    emit({"stage": "model", "progress": 0, "seconds": len(wav) / 16000})
    r = TR.transcribe(wav, langs, simplified, lambda p: emit({"stage": "model", "progress": p}),
                      lambda lines: emit({"draft": lines}))
    emit({"stage": "beam", "lang": r["lang"]})
    r["timing"]["total"] = time.time() - t0
    return r


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))

    def do_GET(self):
        if urlparse(self.path).path not in ("/", "/index.html"):
            self.send_error(404); return
        body = open(os.path.join(HERE, "index.html"), "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/transcribe":
            self.send_error(404); return
        q = parse_qs(u.query)
        langs = [l for l in q.get("lang", [""])[0].split(",") if l in LANGS]
        simplified = q.get("hans", ["0"])[0] == "1"
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()
        try:
            emit({"result": transcribe(data, langs, simplified, emit)})
        except Exception as e:  # noqa: BLE001
            emit({"error": str(e)})


if __name__ == "__main__":
    print(f"open http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
