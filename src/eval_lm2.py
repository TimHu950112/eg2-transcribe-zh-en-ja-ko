"""Greedy vs beam + LM, several LM variants, all languages in parallel (no training).

The model's top-8 per slot is computed once on the GPU; decoding then runs one process per
held-out set, each with an LM read into a dict (lm.BackoffLM's per-lookup binary search and
FNV hashing were the slow part of eval_lm.py).

    python src/eval_lm2.py runs/v2/best.pt 150 lm/trans lm/wiki
"""
import math, os, random, sys, time, collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lm import BOS, SEP, fnv64, ctc_beam

BEAM = int(os.environ.get("BEAM", "16"))


class DictLM:
    def __init__(self, path):
        import struct
        data = open(path, "rb").read()
        _, self.n, count, self.unk = struct.unpack("<IIIf", data[4:20])
        rec = np.frombuffer(data[20:], dtype=[("k", "<u8"), ("p", "<f2"), ("b", "<f2")], count=count)
        self.table = dict(zip(rec["k"].tolist(), zip(rec["p"].astype(np.float32).tolist(), rec["b"].astype(np.float32).tolist())))
        self.hashes = {}
        self.cache = {}

    def _get(self, g):
        h = self.hashes.get(g)
        if h is None:
            h = self.hashes[g] = fnv64(SEP.join(g))
        return self.table.get(h)

    def logp(self, hist, w):
        h = tuple(hist[-(self.n - 1):])
        if len(h) < self.n - 1:
            h = (BOS,) * (self.n - 1 - len(h)) + h
        key = (h, w)
        v = self.cache.get(key)
        if v is not None:
            return v
        bo, out = 0.0, self.unk
        for k in range(len(h), -1, -1):
            hh = h[len(h) - k:]
            e = self._get(hh + (w,))
            if e is not None:
                out = bo + e[0]; break
            if k > 0:
                e = self._get(hh)
                if e is not None:
                    bo += e[1]
        self.cache[key] = out
        return out


def decode_set(args):
    key, items, itos, variants, from_units_ok = args
    sys.path.insert(0, os.path.dirname(__file__))
    from train import error_rate
    lang = key.split("/")[0]
    vocab = None
    if from_units_ok:
        from units import HybridVocab
        vocab = HybridVocab(itos[1:])
    expand = (lambda u: vocab.from_units([u])) if vocab else None
    conv = (lambda h: vocab.from_units(h)) if vocab else (lambda h: h)
    refs = [r for r, _, _ in items]
    out = {}
    # greedy
    hyps = []
    for _, ids, _ in items:
        prev, h = -1, []
        for i in ids[:, 0]:
            if i != prev and i != 0:
                h.append(itos[i])
            prev = i
        hyps.append(conv(h))
    out["greedy"] = error_rate(refs, hyps, [lang] * len(refs))[lang]
    for name, path, grid in variants:
        lm = DictLM(f"{path}/lm_{lang}.bin")
        half = len(items) // 2
        best = None
        for a, b in grid:
            hy = [conv(ctc_beam(ids, lps, itos, lm, alpha=a, beta=b, expand=expand, beam=BEAM)) for _, ids, lps in items[:half]]
            e = error_rate(refs[:half], hy, [lang] * half)[lang]
            if best is None or e < best[0]:
                best = (e, a, b)
        _, a, b = best
        hy = [conv(ctc_beam(ids, lps, itos, lm, alpha=a, beta=b, expand=expand, beam=BEAM)) for _, ids, lps in items]
        out[name] = (error_rate(refs, hy, [lang] * len(refs))[lang], a, b)
    return key, out


def main():
    import torch
    from train import Frames, collate, batches
    from reader import Reader
    from text import Vocab
    ck_path, N, lm_dirs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    ck = torch.load(ck_path, map_location="cpu")
    vocab = Vocab.load(ck_path.rsplit("/", 1)[0] + "/vocab.json")
    m = Reader(**ck["cfg"]).to("mps").eval(); m.load_state_dict(ck["model"])
    jobs = []
    t = time.time()
    for l in ["zh", "ja", "ko", "en"]:
        for key, p in [(l, f"feats/fleurs_{l}_validation"), (l + "/talk", f"feats/yodas2_{l}000_00000000")]:
            ds = Frames([p])
            random.seed(1)
            items = random.sample(ds.items, min(N, len(ds.items)))
            data = []
            with torch.no_grad():
                for ids in batches(items, 6000, shuffle=False):
                    x, L, _, st, lg = collate(ds, items, ids, vocab, "mps")
                    lp, OL = m(x, L, torch.zeros_like(st), lg)
                    v, i = lp.topk(8, -1)
                    v, i = v.float().cpu().numpy(), i.cpu().numpy()
                    del lp; torch.mps.empty_cache()
                    for j, k in enumerate(ids):
                        T = int(OL[j]); data.append((items[k]["sym"], i[j, :T], v[j, :T]))
            grid = [(0.3, 1.0), (0.3, 2.0), (0.5, 1.0), (0.5, 2.0), (0.7, 2.0)]
            variants = [(os.path.basename(d.rstrip("/")), d, grid) for d in lm_dirs]
            jobs.append((key, data, vocab.itos, variants, hasattr(vocab, "from_units")))
    print(f"logits ready {time.time()-t:.0f}s", flush=True)
    del m; torch.mps.empty_cache()
    results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for key, out in ex.map(decode_set, jobs):
            results[key] = out
            print(key, {k: (round(v * 100, 1) if isinstance(v, float) else (round(v[0] * 100, 1), v[1], v[2])) for k, v in out.items()}, flush=True)
    print(f"done {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
