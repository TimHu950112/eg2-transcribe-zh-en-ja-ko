"""LiteRT export vs PyTorch on real clips: slot agreement and the transcripts side by side."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "src")
from ai_edge_litert.interpreter import Interpreter
from export_transcriber import build, WINDOW_SAMPLES
from audio_io import load_audio
from text import Vocab, render
from reader import LANGS

ckpt, path = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3 and sys.argv[3] == "qwen":  # export_qwen.py's graph
    from export_qwen import build as build_qwen
    model = build_qwen(ckpt)
else:
    model, _ = build(ckpt)
vocab = Vocab.load(ckpt.rsplit("/", 1)[0] + "/vocab.json")
it = Interpreter(path, num_threads=8); it.allocate_tensors()
ins, outs = it.get_input_details(), it.get_output_details()
agree_all = []
for lang in ["ja", "zh", "ko", "en"]:
    for line in open(f"data/manifests/fleurs_{lang}_validation.jsonl").readlines()[:3]:
        r = json.loads(line)
        wav = load_audio(r["audio"])[:WINDOW_SAMPLES]
        wav = np.pad(wav, (0, WINDOW_SAMPLES - len(wav)))[None].astype(np.float32)
        lg = np.array([LANGS.index(lang) + 1], dtype=np.int32)
        with torch.no_grad():
            _, ti = model(torch.from_numpy(wav), torch.from_numpy(lg))
        for d in ins:
            it.set_tensor(d["index"], wav if d["shape"][-1] == WINDOW_SAMPLES else lg)
        t = time.time(); it.invoke(); dt = time.time() - t
        ids = next(it.get_tensor(d["index"]) for d in outs if d["dtype"] == np.int32)
        a, b = ti[0, :, 0].numpy(), ids[0, :, 0]
        agree_all.append((a == b).mean())
        print(f"[{lang}] {dt*1000:.0f} ms cpu  agreement {(a == b).mean()*100:.1f}%\n  torch : {render(vocab.decode_ctc(a))}\n  litert: {render(vocab.decode_ctc(b))}")
print(f"mean slot agreement {np.mean(agree_all)*100:.2f}%")
