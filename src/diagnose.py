"""Where the reader's errors come from: fit (train vs dev), homophones, spelling."""
import sys, random, json, collections
import torch, jiwer
sys.path.insert(0, "src")
from train import Frames, evaluate, collate, batches, error_rate
from reader import Reader
from text import Vocab, render, score_text
from pypinyin import lazy_pinyin

ck = torch.load(sys.argv[1], map_location="cpu")
vocab = Vocab.load(sys.argv[1].rsplit("/", 1)[0] + "/vocab.json")
dev = "mps"
m = Reader(**ck["cfg"]).to(dev).eval(); m.load_state_dict(ck["model"])

def run(ds, n=400, seed=0):
    random.seed(seed)
    items = random.sample(ds.items, min(n, len(ds.items)))
    refs, hyps, langs = [], [], []
    with torch.no_grad():
        for ids in batches(items, 12000, shuffle=False):
            x, L, _, st, lg = collate(ds, items, ids, vocab, dev)
            lp, OL = m(x, L, torch.zeros_like(st), lg)
            best = lp.argmax(-1).cpu()
            for j, i in enumerate(ids):
                hyps.append(vocab.decode_ctc(best[j, : int(OL[j])])); refs.append(items[i]["sym"]); langs.append(items[i]["lang"])
    return refs, hyps, langs

print("== fit: same model on training data vs held-out (400 utterances each)")
for name, tr, dv in [("fleurs", "feats/fleurs_{l}_train", "feats/fleurs_{l}_validation"),
                     ("yodas", "feats/yodas2_{l}000_00000003", "feats/yodas2_{l}000_00000000")]:
    for l in ["zh", "ja", "ko", "en"]:
        a = error_rate(*run(Frames([tr.format(l=l)])))
        b = error_rate(*run(Frames([dv.format(l=l)])))
        print(f"  {name:6s} {l}: train {a[l]*100:5.1f}%   held-out {b[l]*100:5.1f}%")

print("== zh FLEURS dev: what kind of character errors")
refs, hyps, langs = run(Frames(["feats/fleurs_zh_validation"]), n=409)
same = diff = ins = dele = 0
for r, h in zip(refs, hyps):
    r, h = score_text(r).replace(" ", ""), score_text(h).replace(" ", "")
    out = jiwer.process_characters(r, h)
    for a in out.alignments[0]:
        if a.type == "substitute":
            for k in range(a.ref_end_idx - a.ref_start_idx):
                rc, hc = r[a.ref_start_idx + k], h[a.hyp_start_idx + k]
                if lazy_pinyin(rc) == lazy_pinyin(hc): same += 1
                else: diff += 1
        elif a.type == "insert": ins += a.hyp_end_idx - a.hyp_start_idx
        elif a.type == "delete": dele += a.ref_end_idx - a.ref_start_idx
tot = same + diff + ins + dele
print(f"  substitutions same sound {same} ({same/tot*100:.0f}%), different sound {diff} ({diff/tot*100:.0f}%), inserted {ins} ({ins/tot*100:.0f}%), dropped {dele} ({dele/tot*100:.0f}%)")
pin_ref = [" ".join(lazy_pinyin(score_text(r).replace(' ', ''))) for r in refs]
pin_hyp = [" ".join(lazy_pinyin(score_text(h).replace(' ', ''))) for h in hyps]
print(f"  pinyin (syllable) error rate: {jiwer.wer(pin_ref, pin_hyp)*100:.1f}%  vs character error {jiwer.cer([score_text(r).replace(' ','') for r in refs],[score_text(h).replace(' ','') or '-' for h in hyps])*100:.1f}%")

print("== en FLEURS dev: letters vs words")
refs, hyps, langs = run(Frames(["feats/fleurs_en_validation"]), n=394)
R = [score_text(r) for r in refs]; H = [score_text(h) or "-" for h in hyps]
print(f"  CER {jiwer.cer(R,H)*100:.1f}%  WER {jiwer.wer(R,H)*100:.1f}%")
for r, h in list(zip(R, H))[:4]: print("   ref:", r, "\n   hyp:", h)
