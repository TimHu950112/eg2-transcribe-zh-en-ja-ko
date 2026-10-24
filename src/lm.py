"""A small symbol n-gram language model per language, and CTC prefix beam search with it.

The reader hears syllables well but picks among homophones (覺得 / 決的) without knowing which
reads as language: CTC outputs each slot on its own. A beam keeps several readings alive and
an n-gram over the same symbols the reader writes says which one people actually write.
Punctuation is transparent to the LM (plain-style sources have none); spaces count (they mark
English and Korean words).

    python src/lm.py build --lang zh --order 5 --out lm/zh.json
"""
import argparse, collections, glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from text import normalize, PUNCT, SPACE

BOS = "<s>"


class NGram:
    """Interpolated Kneser-Ney (one discount per order), log10 probs looked up on the fly."""

    def __init__(self, order):
        self.n = order
        self.counts = [collections.Counter() for _ in range(order + 1)]   # counts[k][(h..., w)]
        self.ctx = [collections.Counter() for _ in range(order + 1)]      # total count of history
        self.types = [collections.Counter() for _ in range(order + 1)]    # distinct followers of history
        self.cont = collections.Counter()  # distinct left contexts of w (unigram continuation)
        self.cont_total = 0
        self.D = 0.75
        self.cache = {}

    def add(self, syms):
        s = [BOS] * (self.n - 1) + [x for x in syms if x not in PUNCT]
        for i in range(self.n - 1, len(s)):
            for k in range(1, self.n + 1):
                g = tuple(s[i - k + 1: i + 1])
                if self.counts[k][g] == 0:
                    self.types[k][g[:-1]] += 1
                    if k == 2:
                        self.cont[g[-1]] += 1
                        self.cont_total += 1
                self.counts[k][g] += 1
                self.ctx[k][g[:-1]] += 1

    def prune(self, min_count=2):
        """Drops rare high-order n-grams (keeps the model small); histories stay consistent enough."""
        for k in range(3, self.n + 1):
            self.counts[k] = collections.Counter({g: c for g, c in self.counts[k].items() if c >= min_count})

    def logp(self, hist, w):
        """log10 P(w | last n-1 symbols of hist)."""
        h = tuple(hist[-(self.n - 1):]) if self.n > 1 else ()
        if len(h) < self.n - 1:
            h = (BOS,) * (self.n - 1 - len(h)) + h
        key = (h, w)
        if key in self.cache:
            return self.cache[key]
        p = (self.cont[w] + 0.5) / (self.cont_total + 0.5 * 10000)  # continuation unigram, smoothed
        for k in range(2, self.n + 1):
            hh = h[len(h) - (k - 1):]
            c = self.ctx[k][hh]
            if c == 0:
                continue
            p = max(self.counts[k][hh + (w,)] - self.D, 0) / c + self.D * self.types[k][hh] / c * p
        lp = math.log10(p)
        if len(self.cache) > 2_000_000:
            self.cache.clear()
        self.cache[key] = lp
        return lp


def _word(prefix, syms_of, end=None):
    """The Latin-letter word that ends the prefix (lowercased), or None if it isn't one."""
    out = []
    for j in reversed(prefix):
        for x in reversed(syms_of(j)):
            if x == SPACE:
                return "".join(reversed(out)).lower() if out else None
            if x in PUNCT:
                continue
            if not (x.isascii() and (x.isalpha() or x == "'")):
                return None
            out.append(x)
    return "".join(reversed(out)).lower() if out else None


def ctc_beam(topk_ids, topk_logp, itos, lm=None, alpha=0.5, beta=1.0, beam=16, prune=-12.0, expand=None, lex=None, oov=0.0):
    """CTC prefix beam search. topk_*: (T, K) natural-log probs of the K best units per slot
    (id 0 = blank). Returns the best symbol list. Score = log P_ctc + alpha*ln P_lm + beta*|symbols|.
    `expand`: unit -> the symbols it writes (a Gemma token "§id" -> several letters); the LM and
    the length bonus see symbols, so character and hybrid readers share one LM.
    `lex`, `oov`: a set of known (lowercase) words; every finished Latin word outside it costs
    `oov` (natural log) — spelled-out English that isn't a word ("annalesis") loses to one that is."""
    if expand is None:
        expand = lambda u: [u]
    cache = {}

    def syms_of(i):
        if i not in cache:
            cache[i] = expand(itos[i])
        return cache[i]
    LN10 = math.log(10)
    NEG = -1e30

    def lse(a, b):
        if a < b:
            a, b = b, a
        return a if b <= NEG else a + math.log1p(math.exp(b - a))

    # prefix -> (p_blank, p_nonblank, lm_score, n_symbols)
    beams = {(): (0.0, NEG, 0.0, 0)}
    for t in range(len(topk_ids)):
        ids, lps = topk_ids[t], topk_logp[t]
        best = lps[0]
        nxt = collections.defaultdict(lambda: [NEG, NEG, 0.0, 0])
        for prefix, (pb, pnb, lms, n) in beams.items():
            total = lse(pb, pnb)
            for i, lp in zip(ids, lps):
                i = int(i)
                if lp < best + prune:
                    continue
                if i == 0:
                    e = nxt[prefix]; e[0] = lse(e[0], total + lp); e[2], e[3] = lms, n
                    continue
                s = itos[i]
                last = prefix[-1] if prefix else None
                if last == i:
                    # Repeat without a blank between: same prefix; with one: a new symbol.
                    e = nxt[prefix]; e[1] = lse(e[1], pnb + lp); e[2], e[3] = lms, n
                    newp = prefix + (i,)
                    add = pb + lp
                else:
                    newp = prefix + (i,)
                    add = total + lp
                e = nxt[newp]
                if e[0] == NEG and e[1] == NEG:
                    new = [x for x in syms_of(i) if x not in PUNCT]
                    score = 0.0
                    if lm is not None and new:
                        # The last n-1 symbols that aren't marks (a comma among the last
                        # units must not shorten the context to a sentence start).
                        hist = []
                        for j in reversed(prefix):
                            hist[:0] = [x for x in syms_of(j) if x not in PUNCT]
                            if len(hist) >= lm.n - 1:
                                break
                        hist = hist[-(lm.n - 1):]
                        for x in new:
                            score += lm.logp(hist, x) * LN10
                            hist.append(x)
                    if lex is not None and new and new[0] == SPACE:
                        w = _word(prefix, syms_of)
                        if w is not None and w not in lex:
                            score += oov / alpha
                    e[2] = lms + alpha * score
                    e[3] = n + len(new)
                e[1] = lse(e[1], add)
        scored = sorted(nxt.items(), key=lambda kv: lse(kv[1][0], kv[1][1]) + kv[1][2] + beta * kv[1][3], reverse=True)
        beams = {k: tuple(v) for k, v in scored[:beam]}
    def final(kv):
        sc = lse(kv[1][0], kv[1][1]) + kv[1][2] + beta * kv[1][3]
        if lex is not None:
            w = _word(kv[0], syms_of)
            if w is not None and w not in lex:
                sc += oov
        return sc
    best = max(beams.items(), key=final)
    return [itos[i] for i in best[0]]  # units; the caller turns them into symbols


def corpus(lang, exclude=("validation", "test", "_00000000")):
    """Training transcripts of one language, as the reader's symbols."""
    for f in sorted(glob.glob("feats/*.jsonl")):
        if any(x in f for x in exclude):
            continue
        if "fleurs" not in f and not os.path.exists(f[:-6] + ".done"):
            continue
        for line in open(f):
            r = json.loads(line)
            if r["lang"] == lang:
                yield normalize(r["text"], lang, plain=bool(r.get("style", 0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--lang")
    ap.add_argument("--order", type=int, default=5)
    args = ap.parse_args()
    lm = NGram(args.order)
    n = 0
    for syms in corpus(args.lang):
        lm.add(syms); n += len(syms)
    print(args.lang, "symbols", n, "5-grams", len(lm.counts[args.order]))


# ---- On-device form -------------------------------------------------------------------------
# Interpolated Kneser-Ney rewritten as a backoff model (exactly the same probabilities): every
# kept n-gram stores log10 P(w|h) and, as a history, log10 of its backoff weight
# gamma(h) = D * types(h) / c(h). P(w|h) = P(hw) if hw is kept, else gamma(h) * P(w|h[1:]).
# File: b"JNLM", u32 version, u32 order, u32 count, f32 unknown log10 prob, then `count`
# entries sorted by key: u64 FNV-1a of the symbols joined by U+001F, f16 logp, f16 backoff.

import struct
import numpy as np

SEP = "\x1f"


def fnv64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def export_backoff(lm: NGram, path: str, min_count=(1, 1, 2, 2, 3)):
    """Keeps n-grams of order k seen at least min_count[k-1] times; writes the binary table."""
    entries = {}
    for k in range(1, lm.n + 1):
        for g, c in lm.counts[k].items():
            if c < min_count[k - 1] or BOS in g[-1:]:
                continue
            h = g[:-1]
            entries[g] = [lm.logp(list(h) if h else [], g[-1]) if k > 1 else math.log10((lm.cont[g[-1]] + 0.5) / (lm.cont_total + 5000)), 0.0]
    # Backoff weights for every kept history (and the BOS-padded ones the decoder starts from).
    for k in range(2, lm.n + 1):
        for h, c in lm.ctx[k].items():
            if c == 0:
                continue
            bow = math.log10(lm.D * lm.types[k][h] / c)
            if h in entries:
                entries[h][1] = bow
            elif all(x == BOS for x in h) or h in entries:
                entries[h] = [0.0, bow]
    keys = np.array([fnv64(SEP.join(g)) for g in entries], dtype=np.uint64)
    vals = np.array(list(entries.values()), dtype=np.float16)
    order = np.argsort(keys)
    unk = math.log10(0.5 / (lm.cont_total + 5000))
    with open(path, "wb") as f:
        f.write(b"JNLM" + struct.pack("<IIIf", 1, lm.n, len(keys), unk))
        rec = np.zeros(len(keys), dtype=[("k", "<u8"), ("p", "<f2"), ("b", "<f2")])
        rec["k"], rec["p"], rec["b"] = keys[order], vals[order, 0], vals[order, 1]
        f.write(rec.tobytes())
    return len(keys)


class BackoffLM:
    """Reads the binary table the way the app does (to check it scores like NGram)."""

    def __init__(self, path):
        data = open(path, "rb").read()
        assert data[:4] == b"JNLM"
        _, self.n, count, self.unk = struct.unpack("<IIIf", data[4:20])
        rec = np.frombuffer(data[20:], dtype=[("k", "<u8"), ("p", "<f2"), ("b", "<f2")], count=count)
        self.keys, self.p, self.b = rec["k"], rec["p"].astype(np.float32), rec["b"].astype(np.float32)
        self.cache = {}

    def _find(self, g):
        k = np.uint64(fnv64(SEP.join(g)))
        i = int(np.searchsorted(self.keys, k))
        return i if i < len(self.keys) and self.keys[i] == k else -1

    def logp(self, hist, w):
        h = tuple(hist[-(self.n - 1):])
        if len(h) < self.n - 1:
            h = (BOS,) * (self.n - 1 - len(h)) + h
        key = (h, w)
        if key in self.cache:
            return self.cache[key]
        bo = 0.0
        out = self.unk
        for k in range(len(h), -1, -1):
            hh = h[len(h) - k:]
            i = self._find(hh + (w,))
            if i >= 0:
                out = bo + float(self.p[i]); break
            if k > 0:
                j = self._find(hh)
                if j >= 0:
                    bo += float(self.b[j])
        if len(self.cache) > 2_000_000:
            self.cache.clear()
        self.cache[key] = out
        return out
