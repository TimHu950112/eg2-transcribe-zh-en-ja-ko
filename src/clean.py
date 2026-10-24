"""Finds YouTube subtitle lines that aren't what was said, using a trained reader.

YODAS2 subtitles are human-made but often not verbatim: a summary, a translation, the song
lyrics instead of the talk, or timed to the wrong stretch. Lines the reader disagrees with
wildly (error > 80%, or a length off by more than 2.5x) are listed in data/clean/drop.json and
left out of training. The threshold is loose on purpose: the reader itself is wrong ~30-40%
of the time on such speech, so only lines far beyond that are dropped.

    python src/clean.py runs/v1/step12500.pt
"""
import glob, json, os, sys
import torch, jiwer
sys.path.insert(0, "src")
from train import Frames, collate, batches
from reader import Reader
from text import Vocab, score_text

ck_path = sys.argv[1]
ck = torch.load(ck_path, map_location="cpu")
vocab = Vocab.load(ck_path.rsplit("/", 1)[0] + "/vocab.json")
m = Reader(**ck["cfg"]).to("mps").eval(); m.load_state_dict(ck["model"])
os.makedirs("data/clean", exist_ok=True)
drop, stats = {}, {}
for p in sorted(glob.glob("feats/yodas2_*.done")):
    p = p[:-5]
    ds = Frames([p])
    bad = []
    with torch.no_grad():
        for ids in batches(ds.items, 16000, shuffle=False):
            x, L, _, st, lg = collate(ds, ds.items, ids, vocab, "mps")
            lp, OL = m(x, L, torch.zeros_like(st), lg)
            best = lp.argmax(-1).cpu()
            del lp
            torch.mps.empty_cache()  # or the MPS cache fragments into tens of GB
            for j, i in enumerate(ids):
                r = ds.items[i]
                ref = score_text(r["sym"]); hyp = score_text(vocab.decode_ctc(best[j, : int(OL[j])]))
                if r["lang"] in ("en",):
                    err = jiwer.wer(ref, hyp or "-") if ref else 1.0
                else:
                    a, b = ref.replace(" ", ""), hyp.replace(" ", "")
                    err = jiwer.cer(a, b or "-") if a else 1.0
                ratio = (len(hyp) + 1) / (len(ref) + 1)
                if err > 0.8 or ratio < 0.4 or ratio > 2.5:
                    bad.append(r["off"])
    drop[p] = bad
    stats[p] = (len(bad), len(ds.items))
    print(p, f"drop {len(bad)}/{len(ds.items)} ({len(bad)/max(1,len(ds.items))*100:.0f}%)", flush=True)
json.dump(drop, open("data/clean/drop.json", "w"))
for lang in ["zh", "ja", "ko", "en"]:
    a = sum(v[0] for k, v in stats.items() if f"_{lang}000_" in k); b = sum(v[1] for k, v in stats.items() if f"_{lang}000_" in k)
    print(lang, f"dropped {a}/{b} ({a/max(1,b)*100:.0f}%)")
