"""Recording -> transcript (中文 / English / 日本語 / 한국어), the same way the phone apps do it.

The exported model (models/) takes 30 s of 16 kHz audio and gives the 8 likeliest symbols for
each 20 ms slot. Longer recordings are cut into windows with 1 s of overlap on each side, the
language is guessed from the rough (greedy) reading, and a CTC beam search with that language's
n-gram picks the final characters. Sentences are cut at 。？！, at a pause, or at a long comma.

    python transcribe.py lecture.m4a                    # language guessed
    python transcribe.py meeting.mp3 --lang zh --hans   # Mandarin, shown in Simplified
    python transcribe.py talk.wav --srt > talk.srt

Backend: Core ML (models/JoinTranscribe.mlpackage) on macOS, LiteRT (models/join_transcribe.tflite)
everywhere else; --backend picks one. Any format ffmpeg reads works.
"""
import argparse, collections, json, math, os, subprocess, sys, time, warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eval_lm2 import DictLM  # noqa: E402
from text import render, PUNCT, SPACE  # noqa: E402

SLOT = 320  # samples per 20 ms slot
EDGE = 50   # slots dropped at each inner window edge (1 s)
TAIL = 25   # the last characters come a little after the sound ends


class Transcriber:
    def __init__(self, models=os.path.join(ROOT, "models"), backend="auto", threads=8):
        self.meta = json.load(open(f"{models}/transcribe_meta.json"))
        self.window, self.k = self.meta["window_samples"], self.meta["topk"]
        self.slots = self.window // SLOT
        self.stride = self.window - 2 * EDGE * SLOT
        self.units = [tuple(u) for u in self.meta["units"]]  # id -> the symbols it writes (id 0 = blank)
        self.langs = self.meta["langs"]
        zh_hans = json.load(open(f"{models}/zh_hans.json"))
        self.hans_chars = zh_hans["chars"]
        self.hans_phrases = sorted(zh_hans["phrases"].items(), key=lambda kv: -len(kv[0]))
        self.lms = {l: DictLM(f"{models}/lm_{l}.bin") for l in self.langs if os.path.exists(f"{models}/lm_{l}.bin")}
        if backend == "auto":
            backend = "coreml" if sys.platform == "darwin" and os.path.exists(f"{models}/JoinTranscribe.mlpackage") else "litert"
        self.backend = backend
        if backend == "coreml":
            import coremltools as ct
            model = ct.models.MLModel(f"{models}/JoinTranscribe.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)

            def predict(audio, lang):
                out = model.predict({"audio": audio, "lang": lang})
                return np.asarray(out["ids"])[0].astype(np.int32), np.asarray(out["logp"])[0].astype(np.float32)
        else:
            from ai_edge_litert.interpreter import Interpreter
            it = Interpreter(f"{models}/join_transcribe.tflite", num_threads=threads)
            it.allocate_tensors()
            ins, outs = it.get_input_details(), it.get_output_details()

            def predict(audio, lang):
                for d in ins:
                    it.set_tensor(d["index"], audio if d["shape"][-1] == self.window else lang)
                it.invoke()
                ids = next(it.get_tensor(d["index"]) for d in outs if d["dtype"] == np.int32)
                lps = next(it.get_tensor(d["index"]) for d in outs if d["dtype"] != np.int32)
                return ids[0].astype(np.int32), lps[0].astype(np.float32)
        self.predict = predict

    def run_model(self, wav, lang_code, progress=lambda p: None, draft=lambda ids: None):
        """Windows over the audio -> stitched (T, K) ids and log-probs."""
        total = max(1, int((len(wav) - self.window) / self.stride) + 2)
        ids, lps = [], []
        pos, w = 0, 0
        lg = np.array([lang_code], dtype=np.int32)
        while True:
            chunk = wav[pos:pos + self.window]
            valid = len(chunk)
            if valid == 0 and w > 0:
                break
            audio = np.zeros((1, self.window), dtype=np.float32)
            audio[0, :valid] = chunk
            wi, wl = self.predict(audio, lg)
            usable = min(self.slots, (valid + SLOT - 1) // SLOT + TAIL)
            last = valid < self.window
            a = 0 if w == 0 else EDGE
            b = usable if last else self.slots - EDGE
            if a < b:
                ids.append(wi[a:b]); lps.append(wl[a:b])
            w += 1
            progress(min(0.99, w / total))
            if ids:
                draft(np.concatenate(ids))
            if last:
                break
            pos += self.stride
        if not ids:
            return np.zeros((0, self.k), np.int32), np.zeros((0, self.k), np.float32)
        return np.concatenate(ids), np.concatenate(lps)

    def greedy(self, ids):
        """[(symbol, slot)] from the best unit per slot."""
        out, prev = [], -1
        for t, i in enumerate(ids[:, 0]):
            i = int(i)
            if i != prev and i != 0:
                out += [(s, t) for s in self.units[i]]
            prev = i
        return out

    def guess_language(self, symbols, among):
        han = kana = hangul = latin = 0
        for s, _ in symbols:
            if len(s) != 1:
                continue
            c = ord(s)
            if 0x3040 <= c <= 0x30FF: kana += 1
            elif 0xAC00 <= c <= 0xD7A3: hangul += 1
            elif 0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF: han += 1
            elif 0x41 <= c <= 0x5A or 0x61 <= c <= 0x7A: latin += 1
        scores = {"ko": hangul, "ja": kana * 3 + han * 0.5 if kana else 0, "zh": han - kana * 2, "en": latin / 4}
        cands = [l for l in self.langs if l in among] or ["zh"]
        return max(cands, key=lambda l: scores[l])

    def beam(self, topk_ids, topk_logp, lm, alpha, beta, width=16, prune=-12.0):
        """CTC prefix beam search with the LM over symbols (src/lm.py ctc_beam), keeping the slot
        each unit was first written in, for the sentence times."""
        LN10, NEG = math.log(10), -1e30
        units = self.units

        def lse(a, b):
            if a < b: a, b = b, a
            return a if b <= NEG else a + math.log1p(math.exp(b - a))

        beams = {(): (0.0, NEG, 0.0, 0, ())}  # prefix -> (p_blank, p_nonblank, lm, n_symbols, slots)
        for t in range(len(topk_ids)):
            ids, lps = topk_ids[t], topk_logp[t]
            best = lps[0]
            nxt = collections.defaultdict(lambda: [NEG, NEG, 0.0, 0, ()])
            for prefix, (pb, pnb, lms, n, sl) in beams.items():
                total = lse(pb, pnb)
                for i, lp in zip(ids, lps):
                    i, lp = int(i), float(lp)
                    if lp < best + prune:
                        continue
                    if i == 0:
                        e = nxt[prefix]; e[0] = lse(e[0], total + lp); e[2], e[3], e[4] = lms, n, sl
                        continue
                    if prefix and prefix[-1] == i:
                        e = nxt[prefix]; e[1] = lse(e[1], pnb + lp); e[2], e[3], e[4] = lms, n, sl
                        newp, add = prefix + (i,), pb + lp
                    else:
                        newp, add = prefix + (i,), total + lp
                    e = nxt[newp]
                    if e[0] == NEG and e[1] == NEG:
                        new = [x for x in units[i] if x not in PUNCT]
                        score = 0.0
                        if new:
                            hist = []
                            for j in reversed(prefix):
                                hist[:0] = [x for x in units[j] if x not in PUNCT]
                                if len(hist) >= lm.n - 1:
                                    break
                            hist = hist[-(lm.n - 1):]
                            for x in new:
                                score += lm.logp(hist, x) * LN10
                                hist.append(x)
                        e[2], e[3], e[4] = lms + alpha * score, n + len(new), sl + (t,)
                    e[1] = lse(e[1], add)
            scored = sorted(nxt.items(), key=lambda kv: lse(kv[1][0], kv[1][1]) + kv[1][2] + beta * kv[1][3], reverse=True)
            beams = {k: tuple(v) for k, v in scored[:width]}
        prefix, v = max(beams.items(), key=lambda kv: lse(kv[1][0], kv[1][1]) + kv[1][2] + beta * kv[1][3])
        return [(s, t) for i, t in zip(prefix, v[4]) for s in units[i]]

    def to_hans(self, text):
        for a, b in self.hans_phrases:
            text = text.replace(a, b)
        return "".join(self.hans_chars.get(c, c) for c in text)

    def lines(self, symbols, simplified=False):
        """Sentences: after 。？！, at a pause over 1.2 s, or at a comma once a line runs long."""
        ends = {"<.>", "<?>", "<!>"}
        out, cur, first, last_at = [], [], -1, -1

        def flush():
            nonlocal cur, first
            text = render(cur)
            if text:
                out.append({"start": max(0, first * 20 - 200), "end": last_at * 20 + 300,
                            "text": self.to_hans(text) if simplified else text})
            cur, first = [], -1

        for s, slot in symbols:
            if cur and slot - last_at > 60 and s not in PUNCT:
                flush()
            if not cur:
                if s == SPACE or s in PUNCT:
                    continue
                first = slot
            cur.append(s); last_at = slot
            if s in ends or (s == "<,>" and len(cur) > 90):
                flush()
        if cur:
            flush()
        return out

    def transcribe(self, wav, langs=(), simplified=False, progress=lambda p: None, draft=lambda lines: None):
        """16 kHz mono float32 -> {"lang", "lines": [{"start", "end" (ms), "text"}], "greedy", "timing"}.
        `langs`: one language to force it, several to choose among, none for any."""
        langs = [l for l in langs if l in self.langs]
        code = self.langs.index(langs[0]) + 1 if len(langs) == 1 else 0
        t1 = time.time()
        ids, lps = self.run_model(wav, code, progress, lambda so_far: draft(self.lines(self.greedy(so_far), simplified)))
        t_model = time.time() - t1
        g = self.greedy(ids)
        lang = langs[0] if len(langs) == 1 else self.guess_language(g, set(langs) if langs else set(self.langs))
        t2 = time.time()
        lm = self.lms.get(lang)
        b = self.beam(ids, lps, lm, *self.meta["lm"].get(lang, [0.3, 1.0])) if lm else g
        return {"lang": lang, "seconds": len(wav) / 16000, "lines": self.lines(b, simplified),
                "greedy": self.lines(g, simplified),
                "timing": {"model": t_model, "beam": time.time() - t2}}


def load_audio(path):
    """Any audio/video file -> 16 kHz mono float32 (ffmpeg)."""
    p = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
                        "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise SystemExit(p.stderr.decode(errors="ignore").strip() or f"ffmpeg could not read {path}")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def srt_time(ms):
    h, ms = divmod(int(ms), 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser(description="On-device speech-to-text for 中文 / English / 日本語 / 한국어")
    ap.add_argument("audio")
    ap.add_argument("--lang", default="", help="zh, en, ja or ko (comma-separated to choose among); default: guess")
    ap.add_argument("--hans", action="store_true", help="Chinese in Simplified characters (default Traditional)")
    ap.add_argument("--srt", action="store_true", help="write SRT subtitles")
    ap.add_argument("--json", action="store_true", help="write the whole result as JSON")
    ap.add_argument("--greedy", action="store_true", help="skip the n-gram beam search (faster, less accurate)")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--backend", default="auto", choices=["auto", "coreml", "litert"])
    args = ap.parse_args()

    tr = Transcriber(args.models, args.backend)
    wav = load_audio(args.audio)
    t0 = time.time()
    r = tr.transcribe(wav, [l for l in args.lang.split(",") if l], args.hans,
                      progress=lambda p: print(f"\r{p * 100:3.0f}%", end="", file=sys.stderr, flush=True))
    print(f"\r{r['seconds']:.0f} s of audio in {time.time() - t0:.1f} s ({tr.backend}, {r['lang']})", file=sys.stderr)
    lines = r["greedy"] if args.greedy else r["lines"]
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
    elif args.srt:
        for n, l in enumerate(lines, 1):
            print(f"{n}\n{srt_time(l['start'])} --> {srt_time(l['end'])}\n{l['text']}\n")
    else:
        for l in lines:
            print(l["text"])


if __name__ == "__main__":
    main()
