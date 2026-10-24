"""Greedy vs beam search with each language's n-gram, on held-out speech (no training).

    python src/eval_lm.py runs/v1/step12500.pt
"""
import random, sys, time
import numpy as np, torch
sys.path.insert(0, "src")
from train import Frames, collate, batches, error_rate
from reader import Reader
from text import Vocab
from lm import NGram, ctc_beam, corpus, export_backoff, BackoffLM
import os

ck_path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
ck = torch.load(ck_path, map_location="cpu")
vocab = Vocab.load(ck_path.rsplit("/", 1)[0] + "/vocab.json")
m = Reader(**ck["cfg"]).to("mps").eval(); m.load_state_dict(ck["model"])

sets = {}
for l in ["zh", "ja", "ko", "en"]:
    sets[l] = Frames([f"feats/fleurs_{l}_validation"])
    sets[l + "/talk"] = Frames([f"feats/yodas2_{l}000_00000000"])


def topk(ds, n):
    random.seed(1)
    items = random.sample(ds.items, min(n, len(ds.items)))
    out = []
    with torch.no_grad():
        for ids in batches(items, 6000, shuffle=False):
            x, L, _, st, lg = collate(ds, items, ids, vocab, "mps")
            lp, OL = m(x, L, torch.zeros_like(st), lg)
            v, i = lp.topk(8, -1)
            v, i = v.float().cpu().numpy(), i.cpu().numpy()
            del lp
            torch.mps.empty_cache()  # (42 GB of fragmented cache otherwise, with a 33k vocabulary)
            for j, k in enumerate(ids):
                T = int(OL[j])
                out.append((items[k]["sym"], i[j, :T], v[j, :T]))
    return out


t0 = time.time()
data = {k: topk(ds, N) for k, ds in sets.items()}
print(f"logits ready {time.time()-t0:.0f}s", flush=True)
lms = {}
for l in ["zh", "ja", "ko", "en"]:
    t = time.time()
    lm = NGram(5)
    for syms in corpus(l):
        lm.add(syms)
    # What the app ships: the pruned binary table, read back the way the app reads it.
    os.makedirs("lm", exist_ok=True)
    n = export_backoff(lm, f"lm/{l}.bin")
    lms[l] = BackoffLM(f"lm/{l}.bin")
    print(f"lm {l}: {n} n-grams, {os.path.getsize(f'lm/{l}.bin')//1024} KB ({time.time()-t:.0f}s)", flush=True)


expand = (lambda u: vocab.from_units([u])) if hasattr(vocab, "from_units") else None


def score(key, alpha, beta, use_lm=True, limit=None):
    lang = key.split("/")[0]
    refs, hyps = [], []
    for ref, ids, lps in data[key][:limit]:
        if alpha is None:
            prev, h = -1, []
            for i in ids[:, 0]:
                if i != prev and i != 0:
                    h.append(vocab.itos[i])
                prev = i
            h = vocab.from_units(h) if hasattr(vocab, "from_units") else h
        else:
            h = ctc_beam(ids, lps, vocab.itos, lms[lang] if use_lm else None, alpha=alpha, beta=beta, expand=expand)
            h = vocab.from_units(h) if hasattr(vocab, "from_units") else h
        refs.append(ref); hyps.append(h)
    return error_rate(refs, hyps, [lang] * len(refs))[lang]


# Tune alpha/beta per language on the first half of the read speech, report on everything.
best = {}
for l in ["zh", "ja", "ko", "en"]:
    grid = [(a, b) for a in (0.3, 0.6) for b in (1.0, 2.0)]  # v1 tuning landed here for every language
    res = [(score(l, a, b, limit=N // 2), a, b) for a, b in grid]
    best[l] = min(res)[1:]
    print(f"tune {l}: best alpha,beta {best[l]}  ({min(res)[0]*100:.1f}%)", flush=True)

print("\n            greedy   beam(no LM)   beam+LM")
for key in data:
    l = key.split("/")[0]
    a, b = best[l]
    g = score(key, None, None)
    nb = score(key, 0.0, 0.0, use_lm=False)
    wl = score(key, a, b)
    print(f"{key:10s}  {g*100:6.1f}%    {nb*100:6.1f}%     {wl*100:6.1f}%", flush=True)
