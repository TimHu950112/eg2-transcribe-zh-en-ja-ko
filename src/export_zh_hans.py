"""zh_hans.json: what the app needs to show a transcript in Simplified Chinese.

The reader writes Traditional with Taiwan wording; OpenCC tw2sp turns that into Mainland
Simplified (軟體 -> 软件, 資料庫 -> 数据库). Shipping OpenCC is far more than the job needs, so
this keeps only: each character the reader can write -> its Simplified form, and the phrases
whose conversion differs from converting their characters one by one. The app replaces phrases
longest first, then characters.

    python src/export_zh_hans.py runs/v1/vocab.json export/v1/zh_hans.json
"""
import json, os, sys
from opencc import OpenCC

data = json.load(open(sys.argv[1]))
vocab = [s for s in (data["itos"] if isinstance(data, dict) else data) if len(s) == 1]
cc = OpenCC("tw2sp")
chars = {c: cc.convert(c) for c in vocab}
chars = {k: v for k, v in chars.items() if k != v}
dic = os.path.join(os.path.dirname(__import__("opencc").__file__), "dictionary")
inv = set(vocab)
phrases = {}
for f in ["TWPhrasesRev.txt", "TWVariantsRevPhrases.txt", "TSPhrases.txt"]:
    for line in open(os.path.join(dic, f), encoding="utf-8"):
        k = line.split("\t")[0].strip()
        if len(k) < 2 or not all(c in inv for c in k):
            continue
        v = cc.convert(k)
        if v != "".join(chars.get(c, c) for c in k):
            phrases[k] = v
json.dump({"chars": chars, "phrases": phrases}, open(sys.argv[2], "w"), ensure_ascii=False, separators=(",", ":"))
print(len(chars), "chars", len(phrases), "phrases", os.path.getsize(sys.argv[2]) // 1024, "KB")
