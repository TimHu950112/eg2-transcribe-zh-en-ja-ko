"""LM experiments on stored logits (eval_ft.py with DUMP=...): tune each language's LM weight on
FLEURS validation, score FLEURS test and held-out YODAS2, for one LM folder.

    python src/lm_grid.py logits/v3.pkl lm/big [beam]
"""
import os, pickle, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import eval_final

if __name__ == "__main__":
    jobs = pickle.load(open(sys.argv[1], "rb"))
    lm_dir = sys.argv[2]
    if len(sys.argv) > 3:
        eval_final.BEAM = int(sys.argv[3])
    jobs = [(l, tune, tests, itos, lm_dir) for l, tune, tests, itos, _ in jobs
            if os.path.exists(f"{lm_dir}/lm_{l}.bin")]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for lang, out in ex.map(eval_final.job, jobs):
            w = out.pop("weights")
            for name, (g, wl, n) in out.items():
                print(f"{lm_dir} {lang} {name:24s} greedy {g:5.1f}%  with LM {wl:5.1f}%  ({n})  weights {w}", flush=True)
