"""The whole on-device model in one graph: 30 s of 16 kHz audio (+ language) -> the 8 likeliest
symbols for each 20 ms slot.

The log-mel front end is inside the graph (a strided conv with the windowed DFT as its weights,
then |.|, the mel matrix and log), so iOS and Android feed raw samples and can't drift from the
features the reader was trained on. Only the top 8 per slot leave the model: greedy decoding
needs the first, a language-model beam the rest, and 1500 x 8 is a few KB instead of 1500 x V.

    python src/export_transcriber.py --ckpt runs/v1/best.pt --out export/v1
"""
import argparse, json, math, os, sys

import numpy as np

# RMSNorm squares activations that reach ~300 (90,000 > float16's 65,504): on the CPU path that
# was inf, the norm zeroed every vector and layer one was 17% off. The model casts these to
# float32 itself; Core ML's float16 pass would drop that, so they're kept wide here.
WIDE_OPS = {"pow", "square", "reduce_mean", "reduce_sum", "reduce_sum_square", "sqrt", "rsqrt", "real_div", "topk"}
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor

sys.path.insert(0, os.path.dirname(__file__))
from export_encoder import TraceableEncoder, load_tower
from reader import Reader

SR = 16000
WINDOW_SAMPLES = 30 * SR  # 480000
TOPK = 8


class LogMel(nn.Module):
    """Gemma4AudioFeatureExtractor as tensor ops: pad 160 left, 320-sample Hann frames every 160,
    512-point |rfft|, 128 HTK mel bands (0-8 kHz), log(x + 1e-3)."""

    def __init__(self, fe):
        super().__init__()
        n, hop, nfft = fe.frame_length, fe.hop_length, fe.fft_length
        self.pad_left, self.hop, self.n = n // 2, hop, n
        t = np.arange(n)[:, None]
        k = np.arange(nfft // 2 + 1)[None, :]
        ang = 2 * np.pi * t * k / nfft
        win = fe.window.astype(np.float64)[:, None]
        basis = np.concatenate([np.cos(ang) * win, -np.sin(ang) * win], axis=1).T  # (2*257, 320)
        # The DFT runs 64x smaller and the magnitude is scaled back after the square root: on the
        # device this graph is float16, where re² + im² of a loud frame passes 65504 (inf -> NaN on
        # the CPU path, an empty transcript).
        self.scale = 64.0
        self.register_buffer("basis", torch.tensor(basis / self.scale, dtype=torch.float32)[:, None, :])
        self.register_buffer("mel", torch.tensor(fe.mel_filters, dtype=torch.float32))  # 257 x 128
        self.floor = float(fe.mel_floor)
        self.bins = nfft // 2 + 1

    def forward(self, wav):  # (1, S)
        x = F.pad(wav, (self.pad_left, 0))[:, None, :]
        # The extractor unfolds n+1 samples per frame and drops the last; same frame starts here.
        spec = F.conv1d(x, self.basis, stride=self.hop)  # (1, 514, frames)
        spec = spec[:, :, : self.frames]
        re, im = spec[:, : self.bins], spec[:, self.bins:]
        mag = torch.sqrt(re * re + im * im)
        return torch.log((mag.transpose(1, 2) @ self.mel) * self.scale + self.floor)  # (1, frames, 128)


class Transcriber(nn.Module):
    def __init__(self, tower, reader, fe, samples=WINDOW_SAMPLES):
        super().__init__()
        self.mel = LogMel(fe)
        mel_frames = (samples + fe.frame_length // 2 - (fe.frame_length + 1)) // fe.hop_length + 1
        self.mel.frames = mel_frames
        self.enc = TraceableEncoder(tower, mel_frames=mel_frames)
        self.T = (mel_frames + 3) // 4
        self.reader = reader
        self.register_buffer("style", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, wav, lang):
        frames = self.enc(self.mel(wav))
        lp = self.reader.forward_static(frames, self.T, self.style, lang.long())
        v, i = lp.topk(TOPK, dim=-1)
        return v, i.to(torch.int32)


def build(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    reader = Reader(**ck["cfg"]).eval()
    tower = load_tower()
    if ck.get("tower"):  # src/finetune.py: encoder and reader trained together
        full = ck["full"]
        keep = ck.get("enc_layers")
        if keep:
            tower.layers = tower.layers[:keep]
            tower.config.num_hidden_layers = keep
        tower.load_state_dict({k[6:]: v for k, v in full.items() if k.startswith("tower.")})
        reader.load_state_dict({k[7:]: v for k, v in full.items() if k.startswith("reader.")})
    else:
        missing, _ = reader.load_state_dict(ck["model"], strict=False)
        if missing:
            print("not in checkpoint (left at init):", missing)
    fe = AutoFeatureExtractor.from_pretrained("google/embeddinggemma-2")
    return Transcriber(tower, reader, fe).eval(), fe


def check(model, fe):
    """The graph against the training path (feature extractor + transformers encoder + reader)."""
    from transformers import AutoModel
    wav = (np.random.randn(WINDOW_SAMPLES) * 0.05).astype(np.float32)
    f = fe([wav], sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        mel = model.mel(torch.from_numpy(wav)[None])
        ref_mel = f["input_features"][:, : mel.shape[1]]
        print("mel frames", mel.shape[1], ref_mel.shape[1], "max diff", (mel - ref_mel).abs().max().item())
        tower = load_tower()
        ref = tower(f["input_features"], f["input_features_mask"], return_dict=True).last_hidden_state
        out = model.enc(mel)
        print("encoder rel diff", ((out - ref[:, : out.shape[1]]).norm() / ref.norm()).item())


def export_coreml(model, out, bits=8, weights_only=False):
    import coremltools as ct
    import coremltools.optimize as cto
    wav = torch.zeros(1, WINDOW_SAMPLES)
    lang = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        ts = torch.jit.trace(model, (wav, lang))
    inputs = [ct.TensorType(name="audio", shape=(1, WINDOW_SAMPLES), dtype=np.float32),
              ct.TensorType(name="lang", shape=(1,), dtype=np.int32)]
    # float32 out (and the top-k picked in float32): the simulator's Core ML handed back a
    # float16 logp of all zeros, and the beam then saw every candidate as equally likely.
    outputs = [ct.TensorType(name="logp", dtype=np.float32), ct.TensorType(name="ids", dtype=np.int32)]
    # The log-mel front end stays float32: a 320-tap DFT summed in float16 and the log of quiet
    # frames lost ~20% on the CPU path (whose float16 doesn't accumulate wider, unlike the GPU)
    # and every layer after inherited it. It's a sliver of the compute; the rest runs in float16.
    probe = ct.convert(ts, inputs=inputs, outputs=outputs, minimum_deployment_target=ct.target.iOS17,
                       compute_precision=ct.precision.FLOAT32, convert_to="milinternal")
    front = set()
    for op in probe.functions["main"].operations:
        if op.op_type == "conv" and op.inputs["x"].rank == 4:  # the encoder's first (2-D) conv
            break
        front.add(op.name)
    m = ct.convert(ts, inputs=inputs, outputs=outputs, minimum_deployment_target=ct.target.iOS17,
                   compute_precision=ct.transform.FP16ComputePrecision(op_selector=lambda op: op.name not in front and op.op_type not in WIDE_OPS))
    if bits == 8:
        q = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=4096)
        # weights_only: the weights of linear and conv layers, not every large constant: the Qwen
        # graph's attention mask and resampling matrix quantized made the GPU compiler abort.
        m = cto.coreml.linear_quantize_weights(m, cto.coreml.OptimizationConfig(
            global_config=None if weights_only else q, op_type_configs={"linear": q} if weights_only else None))
    elif bits == -8:  # 8-bit palettization (a lookup table per tensor): the Qwen graph's GPU path aborted on linear int8
        m = cto.coreml.palettize_weights(m, cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(nbits=8, mode="kmeans", weight_threshold=4096)))
    elif bits == 4:
        m = cto.coreml.palettize_weights(m, cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(nbits=4, granularity="per_grouped_channel", group_size=16,
                                                        weight_threshold=4096)))
    path = f"{out}/JoinTranscribe.mlpackage"
    m.save(path)
    return path


def export_litert(model, out):
    """Run under .venv-litert (litert-torch pins its own torch)."""
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes
    wav = torch.zeros(1, WINDOW_SAMPLES)
    lang = torch.zeros(1, dtype=torch.int32)  # int32 in the graph: what Android hands it
    m = litert_torch.convert(model, (wav, lang), quant_config=quant_recipes.full_dynamic_recipe())
    path = f"{out}/join_transcribe.tflite"
    m.export(path)
    return path


def write_meta(ckpt_path, out, lm_weights=None):
    """transcribe_meta.json: what the app needs to turn the model's units into text, without
    Gemma's tokenizer: `units[i]` is the list of symbols unit i writes (a character, "▁", a
    mark like "<,>", or the letters of a Gemma token, its leading space as "▁")."""
    ck = torch.load(ckpt_path, map_location="cpu")
    from text import Vocab, SPACE
    from reader import LANGS
    vocab = Vocab.load(os.path.join(os.path.dirname(ckpt_path), "vocab.json"))
    if hasattr(vocab, "from_units"):
        units = [[] if i == 0 else vocab.from_units([u]) for i, u in enumerate(vocab.itos)]
    else:
        units = [[] if i == 0 else [u] for i, u in enumerate(vocab.itos)]
    json.dump({"version": 2, "sample_rate": SR, "window_samples": WINDOW_SAMPLES, "slot_ms": 1000 * 30 // 1500, "topk": TOPK,
               "langs": LANGS, "units": units, "dev": {k: round(v, 4) for k, v in ck.get("dev", {}).items()},
               # LM weight and per-symbol bonus per language, tuned on held-out speech (eval_lm2.py).
               "lm": lm_weights or {}},
              open(f"{out}/transcribe_meta.json", "w"), ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--coreml-bits", type=int, default=8, help="16 (none), 8 or 4")
    ap.add_argument("--target", choices=["coreml", "litert", "meta"], default="coreml")
    ap.add_argument("--lm-weights", default=None, help='JSON like {"zh": [0.5, 2.0], ...}')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    write_meta(args.ckpt, args.out, json.loads(args.lm_weights) if args.lm_weights else None)
    if args.target == "meta":
        sys.exit()
    model, fe = build(args.ckpt)
    if args.check:
        check(model, fe)
    print("wrote", export_coreml(model, args.out, args.coreml_bits) if args.target == "coreml" else export_litert(model, args.out))
