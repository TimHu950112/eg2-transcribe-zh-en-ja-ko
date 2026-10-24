"""The same n-gram as lm.py (interpolated Kneser-Ney, one discount, written as a backoff table),
counted with numpy so it can read tens of millions of characters instead of a few.

lm.py's Counter-of-tuples model needs ~200 bytes per n-gram: fine for the 2M symbols of the
training transcripts, 10+ GB for Wikipedia. Here every order's n-grams are rows of a uint16
matrix (one column per symbol), counted with one sort; probabilities and backoff weights for a
whole order are computed at once. The file it writes is byte-for-byte the format lm.BackoffLM
and the apps read; on the same text it gives the same probabilities as lm.NGram (checked in
__main__ with --check).

Text: the training transcripts (spoken; repeated so they weigh more) plus Wikipedia (written,
CC BY-SA: lecture vocabulary, names and terms the transcripts never say).

    python src/lm_big.py --lang zh --wiki-chars 40000000 --out lm/lm_zh.bin
"""
import argparse, math, os, struct, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lm import BOS, SEP, fnv64, corpus
from text import normalize, PUNCT

D = 0.75


def symbol_stream(sentences, order):
    """All sentences as one id array, each preceded by order-1 BOS ids (id 0)."""
    vocab = {BOS: 0}
    ids = []
    for syms in sentences:
        ids.extend([0] * (order - 1))
        for s in syms:
            if s in PUNCT:
                continue
            i = vocab.get(s)
            if i is None:
                i = vocab[s] = len(vocab)
            ids.append(i)
    if len(vocab) >= 65535:
        raise ValueError("more than 65k distinct symbols")
    itos = [None] * len(vocab)
    for s, i in vocab.items():
        itos[i] = s
    return np.array(ids, dtype=np.uint16), itos


def ngrams(stream, k):
    """Unique k-grams (rows, sorted) and their counts; never a BOS as the predicted symbol."""
    n = len(stream) - k + 1
    rows = np.stack([stream[j: j + n] for j in range(k)], axis=1)
    rows = rows[rows[:, -1] != 0]
    uniq, counts = np.unique(rows, axis=0, return_counts=True)
    return uniq, counts.astype(np.int64)


def _void(a):
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def find_rows(table, rows):
    """Index of each of `rows` in the sorted unique `table` (-1 where absent)."""
    if rows.shape[1] == 0:
        return np.zeros(len(rows), dtype=np.int64)
    # Lexicographic order of uint16 rows = order of big-endian bytes.
    tv = _void(table.astype(">u2"))
    rv = _void(rows.astype(">u2"))
    idx = np.searchsorted(tv, rv)
    idx = np.minimum(idx, len(tv) - 1)
    return np.where(tv[idx] == rv, idx, -1)


def thresholds(counts, order, budget):
    """Per-order minimum counts that keep about `budget` n-grams in all (unigrams always kept,
    the rest split evenly across orders 2..n, the most frequent first; never below 2 for 3+)."""
    out = [1]
    share = max(1, (budget - len(counts[1])) // (order - 1))
    for k in range(2, order + 1):
        c = np.sort(counts[k])[::-1]
        t = int(c[share]) + 1 if len(c) > share else 1
        out.append(max(t, 2 if k >= 3 else 1))
    return tuple(out)


def build(sentences, order=5, min_count=(1, 1, 2, 2, 3), budget=None):
    t = time.time()
    stream, itos = symbol_stream(sentences, order)
    print(f"  {len(stream):,} symbols, {len(itos):,} distinct ({time.time()-t:.0f}s)", flush=True)
    grams, counts = {}, {}
    for k in range(1, order + 1):
        grams[k], counts[k] = ngrams(stream, k)
        print(f"  {k}-grams: {len(grams[k]):,}", flush=True)
    del stream
    # Unigram: continuation counts (distinct left neighbours), smoothed as lm.NGram does.
    cont = np.bincount(grams[2][:, 1], minlength=len(itos)).astype(np.float64)
    cont_total = float(len(grams[2]))
    p = {1: (cont[grams[1][:, 0]] + 0.5) / (cont_total + 0.5 * 10000)}
    bow = {}
    for k in range(2, order + 1):
        g, c = grams[k], counts[k]
        hist, inv = np.unique(g[:, :-1], axis=0, return_inverse=True)
        inv = inv.ravel()
        c_h = np.bincount(inv, weights=c).astype(np.float64)
        types_h = np.bincount(inv).astype(np.float64)
        lower = find_rows(grams[k - 1], g[:, 1:])
        p_lower = p[k - 1][lower]
        p[k] = np.maximum(c - D, 0) / c_h[inv] + D * types_h[inv] / c_h[inv] * p_lower
        bow[k] = (hist, np.log10(D * types_h / c_h))
    if budget:
        min_count = thresholds(counts, order, budget)
        print(f"  min counts per order for ~{budget:,} entries: {min_count}", flush=True)
    # Entries: kept n-grams with log10 P; histories carry their backoff weight.
    entries = {}
    for k in range(1, order + 1):
        keep = counts[k] >= min_count[k - 1]
        for row, lp in zip(grams[k][keep], np.log10(p[k][keep])):
            entries[tuple(int(x) for x in row)] = [float(lp), 0.0]
    for k in range(2, order + 1):
        hist, lb = bow[k]
        for row, b in zip(hist, lb):
            key = tuple(int(x) for x in row)
            if key in entries:
                entries[key][1] = float(b)
            elif all(x == 0 for x in key):
                entries[key] = [0.0, float(b)]
    unk = math.log10(0.5 / (cont_total + 5000))
    print(f"  {len(entries):,} entries ({time.time()-t:.0f}s)", flush=True)
    return entries, itos, unk, order


def write(entries, itos, unk, order, path):
    keys = np.array([fnv64(SEP.join(itos[i] for i in g)) for g in entries], dtype=np.uint64)
    vals = np.array(list(entries.values()), dtype=np.float16)
    o = np.argsort(keys)
    rec = np.zeros(len(keys), dtype=[("k", "<u8"), ("p", "<f2"), ("b", "<f2")])
    rec["k"], rec["p"], rec["b"] = keys[o], vals[o, 0], vals[o, 1]
    with open(path, "wb") as f:
        f.write(b"JNLM" + struct.pack("<IIIf", 1, order, len(keys), unk))
        f.write(rec.tobytes())


def wiki_sentences(lang, max_chars):
    """Wikipedia paragraphs as symbol lists (headings and very short lines left out)."""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(f"data/wiki/{lang}.parquet")
    total = 0
    for batch in f.iter_batches(columns=["text"], batch_size=2000):
        for text in batch.column(0).to_pylist():
            for para in text.split("\n"):
                para = para.strip()
                if len(para) < 20:
                    continue
                syms = normalize(para, lang)
                if syms:
                    yield syms
                    total += len(syms)
                    if total >= max_chars:
                        return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--wiki-chars", type=int, default=40_000_000)
    ap.add_argument("--transcript-weight", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1_200_000, help="n-grams kept (~12 bytes each on the device)")
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--talk-chars", type=int, default=0, help="YODAS2 subtitle text (src/yodas_text.py): spoken style")
    ap.add_argument("--talk-weight", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="compare with lm.NGram on the transcripts alone")
    args = ap.parse_args()
    if args.check:
        from lm import NGram
        sents = list(corpus(args.lang))[:20000]
        ref = NGram(5)
        for s in sents:
            ref.add(s)
        entries, itos, unk, order = build(sents, min_count=(1, 1, 1, 1, 1))
        stoi = {s: i for i, s in enumerate(itos)}
        diffs = []
        # Full-order entries hold P(w | 4 symbols) and must equal lm.NGram's; lower orders hold
        # the lower-order distributions used when backing off (lm.export_backoff got those
        # wrong: it scored them as if after a sentence start).
        for g, (lp, _) in entries.items():
            if len(g) != order:
                continue
            diffs.append(abs(lp - ref.logp([itos[i] for i in g[:-1]], itos[g[-1]])))
            if len(diffs) == 20000:
                break
        print(f"check: max |log10 P diff| vs lm.NGram over {len(diffs)} 5-grams = {max(diffs):.2e}")
        sys.exit()
    t = time.time()
    sents = []
    trans = list(corpus(args.lang))
    for _ in range(args.transcript_weight):
        sents.extend(trans)
    n_talk = 0
    if args.talk_chars and os.path.exists(f"data/lmtext/{args.lang}.txt"):
        talk = []
        for line in open(f"data/lmtext/{args.lang}.txt"):
            syms = normalize(line.strip(), args.lang)
            if syms:
                talk.append(syms); n_talk += len(syms)
                if n_talk >= args.talk_chars:
                    break
        for _ in range(args.talk_weight):
            sents.extend(talk)
        print(f"  subtitles {n_talk:,} symbols x{args.talk_weight}", flush=True)
    n_wiki = 0
    for s in wiki_sentences(args.lang, args.wiki_chars):
        sents.append(s); n_wiki += len(s)
    print(f"{args.lang}: transcripts {sum(map(len, trans)):,} symbols x{args.transcript_weight}, wikipedia {n_wiki:,} ({time.time()-t:.0f}s)", flush=True)
    entries, itos, unk, order = build(sents, order=args.order, min_count=(1, 1) + (2,) * (args.order - 2), budget=args.budget)
    write(entries, itos, unk, order, args.out)
    print(f"wrote {args.out}: {os.path.getsize(args.out)/1e6:.1f} MB ({time.time()-t:.0f}s)")
