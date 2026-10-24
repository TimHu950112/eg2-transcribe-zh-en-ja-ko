"""YODAS2 (CC BY 3.0, YouTube with human-made subtitles) shard -> 16 kHz utterance wavs + manifest.

Subtitle lines are short and loosely timed, so consecutive lines are merged into utterances of
up to ~20 s (more context for the reader, fewer cut words at the edges). Lines that aren't speech
in the shard's language are dropped: song lyrics (♪, lots of repeats), captions like [音楽],
and lines whose script doesn't match (a Chinese shard's Japanese line).

    python src/prepare_yodas2.py --tar data/yodas2/ja000/00000000.tar.gz --text data/yodas2/ja000/00000000.json --lang ja
"""
import argparse, html, json, os, re, shutil, tarfile, tempfile

import numpy as np
import soundfile as sf
import librosa

from text import is_cjk, is_hangul

MAX_SEC, MAX_GAP = 20.0, 1.0
_NOISE = re.compile(r"[♪♫♬♩\[\]【】()（）<>《》{}]|https?://|www\.|\.com|@|#")


def script_ok(t: str, lang: str) -> bool:
    letters = [c for c in t if c.isalnum()]
    if len(letters) < 2:
        return False
    han = sum(0x4E00 <= ord(c) <= 0x9FFF for c in letters)
    kana = sum(0x3040 <= ord(c) <= 0x30FF for c in letters)
    hangul = sum(is_hangul(c) for c in letters)
    latin = sum(c.isascii() and c.isalpha() for c in letters)
    n = len(letters)
    if lang == "zh":
        return kana == 0 and hangul == 0 and han / n >= 0.6
    if lang == "ja":
        return hangul == 0 and (kana + han) / n >= 0.6 and kana > 0
    if lang == "ko":
        return hangul / n >= 0.6
    if lang == "en":
        return latin / n >= 0.9
    return False


def segments(text_map):
    """{id: text} with ids '<vid>-<n>-<start cs>-<end cs>' -> sorted (start s, end s, text)."""
    out = []
    for k, t in text_map.items():
        p = k.rsplit("-", 3)
        try:
            s, e = int(p[-2]) / 100.0, int(p[-1]) / 100.0
        except ValueError:
            continue
        t = html.unescape(t).replace("　", " ").strip()
        out.append((s, e, t))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", default="data/yodas2_wav")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    shard = os.path.basename(os.path.dirname(args.tar)) + "_" + os.path.basename(args.tar).split(".")[0]
    wdir = os.path.join(args.out, shard)
    os.makedirs(wdir, exist_ok=True)
    man = args.manifest or f"data/manifests/yodas2_{shard}.jsonl"
    texts = {v["audio_id"]: v["text"] for v in json.load(open(args.text))}
    tmp = tempfile.mkdtemp(dir=os.path.dirname(args.tar))
    kept = dropped = 0
    secs = 0.0
    with tarfile.open(args.tar) as tf, open(man, "w") as fm:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".wav"):
                continue
            vid = os.path.basename(m.name)[:-4]
            if vid not in texts:
                continue
            segs = segments(texts[vid])
            # Lyrics: the same line again and again.
            lines = [t for _, _, t in segs]
            if len(lines) >= 6 and len(set(lines)) / len(lines) < 0.7:
                dropped += len(segs); continue
            good = [(s, e, t) for s, e, t in segs if not _NOISE.search(t) and script_ok(t, args.lang) and e - s >= 0.5]
            dropped += len(segs) - len(good)
            if not good:
                continue
            tf.extract(m, tmp, filter="data")
            path = os.path.join(tmp, m.name)
            try:
                wav, sr = sf.read(path, dtype="float32", always_2d=True)
            except Exception:
                os.remove(path); continue
            wav = wav.mean(1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            os.remove(path)
            # Merge consecutive lines into utterances.
            groups, cur = [], []
            for s, e, t in good:
                if cur and (s - cur[-1][1] > MAX_GAP or e - cur[0][0] > MAX_SEC):
                    groups.append(cur); cur = []
                cur.append((s, e, t))
            if cur:
                groups.append(cur)
            for gi, g in enumerate(groups):
                s, e = max(0.0, g[0][0] - 0.1), g[-1][1] + 0.1
                clip = wav[int(s * 16000): int(e * 16000)]
                if len(clip) < 8000:
                    continue
                sep = " " if args.lang in ("en", "ko") else " "
                text = sep.join(t for _, _, t in g)
                has_punct = bool(re.search(r"[，。？！、,.?!]", text))
                p = os.path.join(wdir, f"{vid}_{gi}.wav")
                sf.write(p, clip, 16000, subtype="PCM_16")
                fm.write(json.dumps({"audio": p, "text": text, "lang": args.lang, "style": 0 if has_punct else 1,
                                     "src": "yodas2"}, ensure_ascii=False) + "\n")
                kept += 1; secs += len(clip) / 16000
    shutil.rmtree(tmp, ignore_errors=True)
    print(shard, "utterances", kept, f"{secs/3600:.2f} h", "lines dropped", dropped)


if __name__ == "__main__":
    main()
