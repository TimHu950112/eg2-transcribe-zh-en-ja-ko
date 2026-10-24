"""Runs the frozen EmbeddingGemma 2 audio encoder over a corpus once and stores its frames.

The encoder never changes while the reader trains, so its output is computed here once:
one int8 matrix per split (all utterances end to end, 25 frames a second, each frame scaled
by its own absmax — half the disk of float16 at ~0.5% error, which SpecAugment dwarfs) plus a
JSONL of (offset, length, scale offset, text, lang). Training reads frames straight from disk.

    python src/extract.py --manifest data/manifests/fleurs_ja_train.jsonl --out feats/fleurs_ja_train
"""
import argparse, io, json, os, time

import numpy as np
import soundfile as sf
import torch
import librosa
from transformers import AutoModel, AutoFeatureExtractor

from audio_io import load_audio

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL: {audio: path|parquet ref, text, lang}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=12, help="encoder layers to keep (12 = all)")
    ap.add_argument("--batch-seconds", type=float, default=240.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    full = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32)
    tower = full.audio_tower
    tower.config.num_hidden_layers = args.layers
    tower = tower.to(dev).eval().half()
    del full
    fe = AutoFeatureExtractor.from_pretrained("google/embeddinggemma-2")

    rows = [json.loads(l) for l in open(args.manifest)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bin_path, meta_path, sc_path = args.out + ".i8", args.out + ".jsonl", args.out + ".scale"
    done = 0
    if os.path.exists(meta_path):  # resume
        done = sum(1 for _ in open(meta_path))
    fbin = open(bin_path, "ab")
    fmeta = open(meta_path, "a")
    fsc = open(sc_path, "ab")
    offset = os.path.getsize(bin_path)
    soff = os.path.getsize(sc_path) // 4

    t0, secs = time.time(), 0.0
    batch = []

    def flush():
        nonlocal offset, secs, soff
        if not batch:
            return
        wavs = [b[1] for b in batch]
        f = fe(wavs, sampling_rate=SR, return_tensors="pt", padding="longest")
        with torch.no_grad():
            o = tower(f["input_features"].to(dev).half(), f["input_features_mask"].to(dev), return_dict=True)
        h = o.last_hidden_state.float().cpu().numpy()
        m = o.attention_mask.cpu().numpy()
        dim = h.shape[-1]
        for (row, wav), hi, mi in zip(batch, h, m):
            n = int(mi.sum())
            sc = np.maximum(np.abs(hi[:n]).max(1), 1e-6).astype(np.float32) / 127.0
            fbin.write(np.round(hi[:n] / sc[:, None]).astype(np.int8).tobytes())
            fsc.write(sc.tobytes())
            fmeta.write(json.dumps({"off": offset, "soff": soff, "n": n, "dim": dim, "text": row["text"],
                                    "lang": row["lang"], "dur": round(len(wav) / SR, 2)}, ensure_ascii=False) + "\n")
            offset += n * dim; soff += n
            secs += len(wav) / SR
        batch.clear()

    cur = 0.0
    for i, row in enumerate(rows[done:], start=done):
        try:
            wav = load_audio(row["audio"], SR)
        except Exception as e:  # a broken file is skipped, not fatal
            print("skip", i, e)
            continue
        if len(wav) > args.max_seconds * SR or len(wav) < 0.3 * SR:
            continue
        batch.append((row, wav))
        cur = max(cur, len(wav) / SR)
        if cur * len(batch) >= args.batch_seconds:
            flush(); cur = 0.0
        if i % 500 == 0:
            el = time.time() - t0
            print(f"{i}/{len(rows)}  {secs/3600:.2f} h audio  {secs/max(el,1e-6):.0f}x realtime", flush=True)
    flush()
    fbin.close(); fmeta.close(); fsc.close()
    print("done", args.out, f"{secs/3600:.2f} h")


if __name__ == "__main__":
    main()
