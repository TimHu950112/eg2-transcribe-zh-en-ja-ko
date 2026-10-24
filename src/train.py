"""Trains the reader on stored encoder frames (src/extract.py) with CTC.

    python src/train.py --train feats/fleurs_*_train --dev feats/fleurs_*_validation --out runs/pilot
"""
import argparse, glob, json, math, os, random, time

import numpy as np
import torch
import jiwer

from reader import Reader, spec_augment, LANGS
from text import Vocab, normalize, render, score_text, is_cjk, is_hangul


class Frames:
    """All utterances of one or more feature files; frames are memory-mapped, not loaded."""

    def __init__(self, prefixes, max_frames=750, drop=None):
        """`drop`: {prefix: [offsets]} of lines left out (src/clean.py)."""
        self.items = []
        self.mm = {}
        for p in prefixes:
            skip = set((drop or {}).get(p, []))
            self.mm[p] = (np.memmap(p + ".i8", dtype=np.int8, mode="r"), np.memmap(p + ".scale", dtype=np.float32, mode="r"))
            for l in open(p + ".jsonl"):
                r = json.loads(l)
                if r["n"] > max_frames or r["off"] in skip:
                    continue
                r["src"] = p
                r["style"] = int(r.get("style", 0))
                # Subtitles typed in ALL CAPS teach nothing about case but how to shout: they
                # count as plain style (recognition only), or the reader capitalises at random.
                letters = [ch for ch in r["text"] if ch.isascii() and ch.isalpha()]
                if len(letters) >= 8 and sum(ch.isupper() for ch in letters) > 0.7 * len(letters):
                    r["style"] = 1
                r["sym"] = normalize(r["text"], r["lang"], plain=bool(r["style"]))
                if not r["sym"]:
                    continue
                self.items.append(r)

    def get(self, r):
        q, sc = self.mm[r["src"]]
        a = np.asarray(q[r["off"]: r["off"] + r["n"] * r["dim"]], dtype=np.float32).reshape(r["n"], r["dim"])
        return torch.from_numpy(a * sc[r["soff"]: r["soff"] + r["n"], None])


def batches(items, max_frames, shuffle=True):
    """Length-bucketed batches of ~max_frames padded frames each."""
    idx = sorted(range(len(items)), key=lambda i: items[i]["n"] + (random.random() * 20 if shuffle else 0))
    out, cur, longest = [], [], 0
    for i in idx:
        n = items[i]["n"]
        if cur and max(longest, n) * (len(cur) + 1) > max_frames:
            out.append(cur); cur, longest = [], 0
        cur.append(i); longest = max(longest, n)
    if cur:
        out.append(cur)
    if shuffle:
        random.shuffle(out)
    return out


def collate(ds, items, ids, vocab, dev, hide_lang=0.0):
    feats = [ds.get(items[i]) for i in ids]
    L = torch.tensor([f.shape[0] for f in feats])
    # Padded to a multiple of 64 frames: a handful of shapes instead of one per batch, or the MPS
    # allocator keeps a cache for every one of them (24 GB in a few hundred steps).
    T = -(-int(L.max()) // 64) * 64
    x = torch.zeros(len(feats), T, feats[0].shape[1])
    for j, f in enumerate(feats):
        x[j, : f.shape[0]] = f
    tg = [torch.tensor(vocab.encode_item(items[i]), dtype=torch.long) for i in ids]
    st = torch.tensor([items[i]["style"] for i in ids])
    lg = torch.tensor([LANGS.index(items[i]["lang"]) + 1 for i in ids])
    if hide_lang:
        lg = torch.where(torch.rand(len(ids)) < hide_lang, torch.zeros_like(lg), lg)
    return x.to(dev), L.to(dev), tg, st.to(dev), lg.to(dev)


def error_rate(refs, hyps, langs):
    """CER for zh/ja/ko (per character, spaces ignored), WER for en — per language."""
    res = {}
    for lang in sorted(set(langs)):
        r = [score_text(a) for a, l in zip(refs, langs) if l == lang]
        h = [score_text(b) for b, l in zip(hyps, langs) if l == lang]
        if lang.startswith("en"):
            res[lang] = jiwer.wer(r, [x if x.strip() else "-" for x in h])
        else:
            r = [x.replace(" ", "") for x in r]
            h = [x.replace(" ", "") or "-" for x in h]
            res[lang] = jiwer.cer(r, h)
    return res


@torch.no_grad()
def evaluate(model, ds, vocab, dev, max_frames=12000, show=2, known_lang=True):
    model.eval()
    refs, hyps, langs = [], [], []
    for ids in batches(ds.items, max_frames, shuffle=False):
        x, L, _, st, lg = collate(ds, ds.items, ids, vocab, dev, hide_lang=0.0 if known_lang else 1.0)
        st = torch.zeros_like(st)  # always judged on the written style the app uses
        lp, OL = model(x, L, st, lg)
        best = lp.argmax(-1).cpu()
        del lp
        if dev == "mps":
            torch.mps.empty_cache()  # same fragmentation as in training (41 GB with a 24k vocabulary)
        for j, i in enumerate(ids):
            hyps.append(vocab.decode_ctc(best[j, : int(OL[j])]))
            refs.append(ds.items[i]["sym"])
            # Reported per language and per kind of speech: read (FLEURS) vs spontaneous (YODAS2).
            langs.append(ds.items[i]["lang"] + ("/talk" if "yodas" in ds.items[i]["src"] else ""))
    model.train()
    shown = {}
    for r, h, l in zip(refs, hyps, langs):
        if shown.get(l, 0) < show:
            shown[l] = shown.get(l, 0) + 1
            print(f"  [{l}] ref: {render(r)}\n       hyp: {render(h)}")
    return error_rate(refs, hyps, langs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default=None, help="existing vocab.json (else built from train)")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-frames", type=int, default=24000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--init", default=None)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=2048)
    ap.add_argument("--drop", default=None, help="data/clean/drop.json: training lines to leave out")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--up", type=int, default=2, help="output slots per encoder frame")
    ap.add_argument("--unit", choices=["char", "gemma", "hybrid"], default="char")
    ap.add_argument("--head-from-gemma", action="store_true", help="start the output layer from Gemma's token embeddings")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    expand = lambda ps: sorted({q[: -len(".jsonl")] for p in ps for q in glob.glob(p + ".jsonl")})
    drop = json.load(open(args.drop)) if args.drop else None
    tr, dv = Frames(expand(args.train), drop=drop), Frames(expand(args.dev))
    print("train", len(tr.items), "dev", len(dv.items))
    if args.vocab:
        vocab = Vocab.load(args.vocab)
    elif args.unit == "gemma":
        from units import GemmaVocab
        vocab = GemmaVocab.build(tr.items)
    elif args.unit == "hybrid":
        from units import HybridVocab
        vocab = HybridVocab.build(tr.items)
    else:
        syms = {}
        for r in tr.items:
            for s in r["sym"]:
                syms[s] = syms.get(s, 0) + 1
        vocab = Vocab(sorted(syms, key=lambda s: -syms[s]))
    vocab.save(f"{args.out}/vocab.json")
    print("vocab", len(vocab))

    model = Reader(len(vocab), in_dim=tr.items[0]["dim"], d=args.d, layers=args.layers, heads=args.heads, ff=args.ff, up=args.up).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev)["model"], strict=False)
    if args.head_from_gemma and args.d != 512:
        print("head-from-gemma skipped: Gemma's token embeddings are 512-d, this reader's output is", args.d)
    elif args.head_from_gemma:
        from transformers import AutoModel
        E = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32).language_model.embed_tokens.weight.detach()
        with torch.no_grad():
            # Rows for Gemma-token units start at that token's embedding (scaled like the rest).
            if hasattr(vocab, "gids"):
                pos, gid = list(range(2, len(vocab))), vocab.gids[2:]
            else:
                pairs = [(k, int(u[1:])) for k, u in enumerate(vocab.itos) if u.startswith("§")]
                pos, gid = [p for p, _ in pairs], [g for _, g in pairs]
            rows = E[torch.tensor(gid)].to(dev)
            scale = model.head.weight.norm(dim=1).mean()
            model.head.weight[torch.tensor(pos, device=dev)] = rows / rows.norm(dim=1, keepdim=True) * scale
        del E
    print("reader params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    step, best, t0, run, skipped = 0, 1e9, time.time(), 0.0, 0
    while step < args.steps:
        for ids in batches(tr.items, args.batch_frames):
            x, L, tg, st, lg = collate(tr, tr.items, ids, vocab, dev, hide_lang=0.3)
            x = spec_augment(x, L)
            with torch.autocast(dev, dtype=torch.bfloat16):
                lp, OL = model(x, L, st, lg)
            tl = torch.tensor([len(t) for t in tg])
            # On the GPU: with a 33k-unit vocabulary, moving the log-probs to the CPU for CTC took
            # 4 s a step (0.2 s here, same loss).
            loss = ctc(lp.transpose(0, 1).float(), torch.cat(tg).to(dev), OL, tl.to(dev))
            opt.zero_grad()
            # One bad batch (CTC on the GPU can return NaN for a long character target) must not
            # poison the weights: such a batch is skipped, and counted.
            if not torch.isfinite(loss):
                skipped += 1; step += 1; sched.step()
                continue
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                opt.zero_grad(); skipped += 1; step += 1; sched.step()
                continue
            opt.step(); sched.step(); step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if dev == "mps":
                # Live memory stays ~5 GB, but the MPS cache fragments by ~1 GB a step without this.
                torch.mps.empty_cache()
            if step % 100 == 0:
                print(f"step {step} loss {run:.3f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s skipped {skipped}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                er = evaluate(model, dv, vocab, dev)
                avg = sum(er.values()) / len(er)  # every language and kind of speech counts the same
                print(f"== step {step} dev", {k: round(v * 100, 2) for k, v in er.items()}, f"avg {avg*100:.2f}", flush=True)
                ea = evaluate(model, dv, vocab, dev, show=0, known_lang=False)
                print(f"   (language unknown)", {k: round(v * 100, 2) for k, v in ea.items()}, flush=True)
                ck = {"model": model.state_dict(), "cfg": model.cfg, "step": step, "dev": er}
                torch.save(ck, f"{args.out}/last.pt")
                if avg < best:
                    best = avg; torch.save(ck, f"{args.out}/best.pt")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
