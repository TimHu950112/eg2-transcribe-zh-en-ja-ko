"""Fetches corpora shard by shard and turns each into encoder frames, deleting the audio after.

Raw audio for a few hundred hours doesn't fit next to its frames on one laptop disk, so each
shard goes download -> utterance wavs -> frames -> delete. Downloads resume (HF drops long
unauthenticated transfers), and a shard whose frames already exist is skipped, so the whole
thing can be stopped and started again.

    python src/pipeline.py yodas2:zh000:1-15 yodas2:ja000:1-20 zeroth parquet:fixie-ai/common_voice_17_0:ja
"""
import glob, json, os, subprocess, sys, time

import pyarrow.parquet as pq

HF = "https://huggingface.co/datasets"
PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def fetch(url, dest, check=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(50):
        subprocess.run(["curl", "-sL", "--retry", "5", "-y", "30", "-Y", "20000", "-C", "-", "-o", dest, url])
        if check is None or check(dest):
            return True
        time.sleep(5)
    return False


def gz_ok(p):
    return subprocess.run(["gzip", "-t", p], capture_output=True).returncode == 0


def parquet_ok(p):
    try:
        pq.read_metadata(p); return True
    except Exception:
        return False


def extract(manifest, out):
    """One encoder on the GPU at a time (several at once swap the machine to a standstill)."""
    if os.path.exists(out + ".done"):
        return
    import fcntl
    lock = open("feats/.extract.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    subprocess.run([PY, "src/extract.py", "--manifest", manifest, "--out", out], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    open(out + ".done", "w").close()
    lock.close()


def yodas2(subset, lo, hi):
    lang = subset[:2]
    for i in range(lo, hi + 1):
        shard = f"{i:08d}"
        out = f"feats/yodas2_{subset}_{shard}"
        if os.path.exists(out + ".done"):
            continue
        tar, txt = f"data/yodas2/{subset}/{shard}.tar.gz", f"data/yodas2/{subset}/{shard}.json"
        man = f"data/manifests/yodas2_{subset}_{shard}.jsonl"
        if os.path.exists(man) and not os.path.exists(tar):  # cut already, frames not yet
            extract(man, out)
            subprocess.run(["rm", "-rf", f"data/yodas2_wav/{subset}_{shard}"])
            print("done", out, flush=True)
            continue
        if not fetch(f"{HF}/espnet/yodas2/resolve/main/data/{subset}/audio/{shard}.tar.gz", tar, gz_ok):
            print("give up", tar); continue
        fetch(f"{HF}/espnet/yodas2/resolve/main/data/{subset}/text/{shard}.json", txt)
        subprocess.run([PY, "src/prepare_yodas2.py", "--tar", tar, "--text", txt, "--lang", lang, "--manifest", man],
                       check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"})
        os.remove(tar)
        extract(man, out)
        subprocess.run(["rm", "-rf", f"data/yodas2_wav/{subset}_{shard}"])
        print("done", out, flush=True)


def parquet_corpus(name, repo, files, lang, text_col, style, audio_col="audio"):
    """A parquet-hosted corpus: each file -> wavs -> frames."""
    for f in files:
        base = os.path.basename(f).split(".")[0]
        out = f"feats/{name}_{base}"
        if os.path.exists(out + ".done"):
            continue
        dest = f"data/{name}/{os.path.basename(f)}"
        if not fetch(f"{HF}/{repo}/resolve/main/{f}" if not f.startswith("http") else f, dest, parquet_ok):
            print("give up", dest); continue
        wdir = f"data/{name}_wav/{base}"
        os.makedirs(wdir, exist_ok=True)
        man = f"data/manifests/{name}_{base}.jsonl"
        t = pq.read_table(dest, columns=[audio_col, text_col])
        with open(man, "w") as fm:
            for j, r in enumerate(t.to_pylist()):
                a = r[audio_col]
                ext = os.path.splitext(a.get("path") or "x.wav")[1] or ".wav"
                p = f"{wdir}/{j}{ext}"
                open(p, "wb").write(a["bytes"])
                fm.write(json.dumps({"audio": p, "text": r[text_col], "lang": lang, "style": style, "src": name},
                                    ensure_ascii=False) + "\n")
        del t
        os.remove(dest)
        extract(man, out)
        subprocess.run(["rm", "-rf", wdir])
        print("done", out, flush=True)


def aishell():
    """AISHELL-1 (Apache 2.0): read Mandarin, one tar per speaker, transcripts word-spaced, no punctuation."""
    import tarfile
    trans = {}
    for l in open("data/aishell/transcript.txt", encoding="utf-8"):
        k, *w = l.split()
        trans[k] = "".join(w)
    import urllib.request
    tree = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/AISHELL/AISHELL-1/tree/main/data_aishell/wav"))
    spk = [x["path"] for x in tree]
    for g in range(0, len(spk), 20):  # 20 speakers per feature file
        out = f"feats/aishell_{g:03d}"
        if os.path.exists(out + ".done"):
            continue
        wdir = f"data/aishell_wav/{g:03d}"
        os.makedirs(wdir, exist_ok=True)
        man = f"data/manifests/aishell_{g:03d}.jsonl"
        with open(man, "w") as fm:
            for f in spk[g:g + 20]:
                dest = f"data/aishell/{os.path.basename(f)}"
                if not fetch(f"{HF}/AISHELL/AISHELL-1/resolve/main/{f}", dest, gz_ok):
                    continue
                with tarfile.open(dest) as tf:
                    for m in tf:
                        k = os.path.basename(m.name)[:-4]
                        if m.isfile() and m.name.endswith(".wav") and k in trans:
                            m.name = os.path.basename(m.name)
                            tf.extract(m, wdir, filter="data")
                            fm.write(json.dumps({"audio": f"{wdir}/{m.name}", "text": trans[k], "lang": "zh",
                                                 "style": 1, "src": "aishell"}, ensure_ascii=False) + "\n")
                os.remove(dest)
        extract(man, out)
        subprocess.run(["rm", "-rf", wdir])
        print("done", out, flush=True)


def hf_parquet_files(repo, config, split):
    import urllib.request
    d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/parquet"))
    return d[config][split]


def main():
    for job in sys.argv[1:]:
        kind, *rest = job.split(":")
        if kind == "yodas2":
            subset, rng = rest
            lo, hi = map(int, rng.split("-"))
            yodas2(subset, lo, hi)
        elif kind == "zeroth":
            files = [f"data/train-0000{i}-of-00006-{h}.parquet" for i, h in enumerate(
                ["6f62b6637bde6a2a", "a6254d939f218ad2", "b7145cf53a2c538b", "eee3a0e1b02f4f15", "c19f5a576971a222", "bed24b01dd0a0946"])]
            parquet_corpus("zeroth", "Bingsu/zeroth-korean", files, "ko", "text", 1)
        elif kind == "cv":  # Common Voice 17 (CC0) via fixie-ai parquet: cv:<locale>:<lang>
            locale, lang = rest
            for split in ["train", "validation"]:
                parquet_corpus(f"cv_{locale}_{split}", "", hf_parquet_files("fixie-ai/common_voice_17_0", locale, split),
                               lang, "sentence", 0)
        elif kind == "aishell":
            aishell()
        elif kind == "librispeech":  # clean-100 (CC BY 4.0)
            parquet_corpus("libri100", "", hf_parquet_files("openslr/librispeech_asr", "clean", "train.100"), "en", "text", 1)
        else:
            print("unknown job", job)


if __name__ == "__main__":
    main()
