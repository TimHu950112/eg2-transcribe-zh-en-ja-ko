"""Exported model vs PyTorch on real clips: same transcript?"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "src")
from export_transcriber import build, WINDOW_SAMPLES
from audio_io import load_audio
from text import Vocab, render
from reader import LANGS

ckpt, pkg = sys.argv[1], sys.argv[2]
model, _ = build(ckpt)
vocab = Vocab.load(ckpt.rsplit("/", 1)[0] + "/vocab.json")
import coremltools as ct
cm = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_GPU)

def greedy(ids):
    return render(vocab.decode_ctc(ids))

for lang in ["ja", "zh", "ko", "en"]:
    r = json.loads(open(f"data/manifests/fleurs_{lang}_validation.jsonl").readline())
    wav = load_audio(r["audio"])[:WINDOW_SAMPLES]
    wav = np.pad(wav, (0, WINDOW_SAMPLES - len(wav)))[None].astype(np.float32)
    lg = np.array([LANGS.index(lang) + 1], dtype=np.int32)
    with torch.no_grad():
        _, ti = model(torch.from_numpy(wav), torch.from_numpy(lg))
    t = time.time(); out = cm.predict({"audio": wav, "lang": lg}); dt = time.time() - t
    a, b = greedy(ti[0, :, 0].numpy()), greedy(out["ids"][0, :, 0])
    agree = (ti[0, :, 0].numpy() == out["ids"][0, :, 0]).mean()
    print(f"[{lang}] {dt*1000:.0f} ms  slot agreement {agree*100:.1f}%\n  torch : {a}\n  coreml: {b}\n  ref   : {r['text']}")
