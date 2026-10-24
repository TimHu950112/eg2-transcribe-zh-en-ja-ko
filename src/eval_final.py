"""Final numbers: LM weights tuned on FLEURS validation, then FLEURS test (never used for any
choice) and the held-out YODAS2 shard (spontaneous speech) scored once, greedy and with the
shipped n-grams. Writes the tuned weights for export (transcribe_meta.json "lm").

    python src/eval_final.py runs/v3/best.pt lm/final > logs/final.txt
"""
import json, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from eval_lm2 import DictLM
from lm import ctc_beam

GRID = [(0.3, 1.0), (0.3, 2.0), (0.5, 1.0), (0.5, 2.0), (0.7, 2.0), (0.5, 3.0), (0.7, 3.0), (1.0, 3.0), (1.0, 4.0)]
BEAM = int(os.environ.get("BEAM", "16"))


def greedy(ids, itos):
    prev, h = -1, []
    for i in ids[:, 0]:
        if i != prev and i != 0:
            h.append(itos[i])
        prev = i
    return h


def job(args):
    lang, tune, tests, itos, lm_dir = args
    from train import error_rate
    lm = DictLM(f"{lm_dir}/lm_{lang}.bin")
    best = None
    for a, b in GRID:
        hy = [ctc_beam(ids, lps, itos, lm, alpha=a, beta=b, beam=BEAM) for _, ids, lps in tune]
        e = error_rate([r for r, _, _ in tune], hy, [lang] * len(tune))[lang]
        if best is None or e < best[0]:
            best = (e, a, b)
    _, a, b = best
    out = {"weights": [a, b]}
    for name, items in tests.items():
        refs = [r for r, _, _ in items]
        g = error_rate(refs, [greedy(ids, itos) for _, ids, _ in items], [lang] * len(items))[lang]
        w = error_rate(refs, [ctc_beam(ids, lps, itos, lm, alpha=a, beta=b, beam=BEAM) for _, ids, lps in items], [lang] * len(items))[lang]
        out[name] = (round(g * 100, 2), round(w * 100, 2), len(items))
    return lang, out


def main():
    import torch
    from train import Frames, collate, batches
    from reader import Reader
    from text import Vocab
    ck_path, lm_dir = sys.argv[1], sys.argv[2]
    ck = torch.load(ck_path, map_location="cpu")
    vocab = Vocab.load(ck_path.rsplit("/", 1)[0] + "/vocab.json")
    m = Reader(**ck["cfg"]).to("mps").eval(); m.load_state_dict(ck["model"])

    def topk(prefix, n=None, seed=1):
        ds = Frames([prefix])
        items = ds.items if n is None else random.Random(seed).sample(ds.items, min(n, len(ds.items)))
        out = []
        # Every batch the same shape (8 x 768 frames): MPS compiles a graph per new shape, and
        # the varied shapes of a whole test set spent 20 minutes compiling, not computing.
        B, TMAX = 8, 768
        with torch.no_grad():
            for start in range(0, len(items), B):
                ids = list(range(start, min(start + B, len(items))))
                feats = [ds.get(items[i]) for i in ids]
                x = torch.zeros(B, TMAX, feats[0].shape[1])
                for j, f in enumerate(feats):
                    x[j, : f.shape[0]] = f
                L = torch.tensor([f.shape[0] for f in feats] + [1] * (B - len(feats)))
                x, L = x.to("mps"), L.to("mps")
                zeros = torch.zeros(B, dtype=torch.long, device="mps")
                lp, OL = m(x, L, zeros, zeros)  # language unknown, as the app's default
                v, i = lp.topk(8, -1)
                v, i = v.float().cpu().numpy(), i.cpu().numpy()
                del lp; torch.mps.empty_cache()
                for j, k in enumerate(ids):
                    T = int(OL[j]); out.append((items[k]["sym"], i[j, :T], v[j, :T]))
        return out

    t = time.time()
    jobs = []
    for lang in ["zh", "ja", "ko", "en"]:
        tune = topk(f"feats/fleurs_{lang}_validation", 120)
        tests = {"read (FLEURS test)": topk(f"feats/fleurs_{lang}_test"),
                 "talk (YODAS2 held out)": topk(f"feats/yodas2_{lang}000_00000000", 300)}
        jobs.append((lang, tune, tests, vocab.itos, lm_dir))
    print(f"logits ready {time.time()-t:.0f}s (language: detected, as the app's default)", flush=True)
    del m; torch.mps.empty_cache()
    weights = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for lang, out in ex.map(job, jobs):
            weights[lang] = out.pop("weights")
            for name, (g, w, n) in out.items():
                unit = "WER" if lang == "en" else "CER"
                print(f"{lang} {name:24s} {unit} greedy {g:5.1f}%  with LM {w:5.1f}%  ({n} utterances)", flush=True)
    json.dump(weights, open(os.path.join(os.path.dirname(ck_path), "lm_weights.json"), "w"))
    print("weights", json.dumps(weights), f"done {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
