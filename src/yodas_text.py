"""Spoken-language text for the n-grams: the human-made subtitles of every YODAS2 shard (the
text files are ~1 MB a shard; the audio of most shards was never fetched). One line per
subtitle line that passes prepare_yodas2's filters. Shard 00000000 (held-out speech) is skipped.

    python src/yodas_text.py zh000:1-41 ja000:1-210 ko000:1-390 en000:1-300
"""
import html, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from prepare_yodas2 import script_ok, _NOISE

HF = "https://huggingface.co/datasets/espnet/yodas2/resolve/main/data"
os.makedirs("data/lmtext", exist_ok=True)


def fetch(subset, i):
    dest = f"data/yodas2/{subset}/{i:08d}.json"
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        subprocess.run(["curl", "-sL", "--retry", "5", "-y", "30", "-Y", "20000", "-o", dest + ".part",
                        f"{HF}/{subset}/text/{i:08d}.json"])
        os.replace(dest + ".part", dest)
    try:
        return json.load(open(dest))
    except Exception:
        return []


def main():
    for job in sys.argv[1:]:
        subset, rng = job.split(":")
        lo, hi = map(int, rng.split("-"))
        lang = subset[:2]
        lines = 0
        with open(f"data/lmtext/{lang}.txt", "a") as out, ThreadPoolExecutor(8) as ex:
            for vids in ex.map(lambda i: fetch(subset, i), range(max(1, lo), hi + 1)):
                for v in vids:
                    for t in v["text"].values():
                        t = html.unescape(t).replace("　", " ").strip()
                        if not t or _NOISE.search(t) or not script_ok(t, lang):
                            continue
                        out.write(t.replace("\n", " ") + "\n"); lines += 1
        print(subset, lines, "lines", flush=True)


if __name__ == "__main__":
    main()
