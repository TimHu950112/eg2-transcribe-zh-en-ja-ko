"""The lexicon's OOV cost on stored logits: English only, LM weights fixed, cost swept.

    python src/lex_grid.py logits/v3.pkl lm/big3 0.3 1.0
"""
import os, pickle, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from eval_lm2 import DictLM
from lm import ctc_beam
from train import error_rate

COSTS = [0.0, -1.0, -2.0, -3.0, -5.0, -8.0]


def run(args):
    oov, items, itos, lm_dir, a, b = args
    lm = DictLM(f"{lm_dir}/lm_en.bin")
    lex = set(open(f"{lm_dir}/lex_en.txt").read().split())
    out = {}
    for name, its in items.items():
        hy = [ctc_beam(ids, lps, itos, lm, alpha=a, beta=b, lex=lex, oov=oov) for _, ids, lps in its]
        out[name] = error_rate([r for r, _, _ in its], hy, ["en"] * len(its))["en"]
    return oov, out


if __name__ == "__main__":
    jobs = pickle.load(open(sys.argv[1], "rb"))
    lm_dir, a, b = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    lang, tune, tests, itos, _ = [j for j in jobs if j[0] == "en"][0]
    items = {"val": tune, **tests}
    with ProcessPoolExecutor(len(COSTS)) as ex:
        for oov, out in ex.map(run, [(c, items, itos, lm_dir, a, b) for c in COSTS]):
            print(f"oov {oov:5.1f}", {k: round(v * 100, 2) for k, v in out.items()}, flush=True)
