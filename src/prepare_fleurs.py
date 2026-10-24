"""FLEURS (CC BY 4.0) parquet -> wav files + manifest JSONL ({audio, text, lang})."""
import json, os, sys
import pyarrow.parquet as pq

LANGS = {"cmn_hans_cn": "zh", "ja_jp": "ja", "ko_kr": "ko", "en_us": "en"}
root = "data/fleurs"
os.makedirs("data/manifests", exist_ok=True)
for code, lang in LANGS.items():
    for split in sys.argv[1:] or ["train", "validation", "test"]:
        src = f"{root}/{code}/{split}.parquet"
        try:
            pq.read_metadata(src)
        except Exception:
            print("not ready:", src); continue
        if os.path.exists(f"data/manifests/fleurs_{lang}_{split}.jsonl"):
            continue
        t = pq.read_table(src, columns=["id", "audio", "raw_transcription"])
        wdir = f"{root}/{code}/{split}_wav"
        os.makedirs(wdir, exist_ok=True)
        seen = 0
        with open(f"data/manifests/fleurs_{lang}_{split}.jsonl", "w") as f:
            for i, r in enumerate(t.to_pylist()):
                p = f"{wdir}/{i}.wav"
                if not os.path.exists(p):
                    open(p, "wb").write(r["audio"]["bytes"])
                f.write(json.dumps({"audio": p, "text": r["raw_transcription"], "lang": lang}, ensure_ascii=False) + "\n")
                seen += 1
        print(code, split, seen)
