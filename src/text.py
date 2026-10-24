"""Text side of the reader: one shared character vocabulary for 中文 / English / 日本語 / 한국어.

Why characters and not the 229k English words of the original: Chinese and Japanese have no
spaces to cut words at, a word list can never hold every name or term in a lecture, and a
character set that covers all four languages is ~8k symbols — small, closed, and code-switching
("這個 function 會 return 一個 array") falls out for free.

Choices that make the output read well:
- Chinese is learned in Traditional characters with Taiwan wording (OpenCC s2twp); a zh-Hans
  reader gets it converted back at display time (t2s), which is near lossless in that direction.
- Punctuation is five abstract symbols (, . ? ! 、). `render` writes them full-width after
  Han / kana and half-width elsewhere, so 「你好，世界。」 and "Hello, world." both come out right.
- Spaces are one symbol, kept for English and Korean, dropped between CJK characters.
"""
import json
import re
import unicodedata

from opencc import OpenCC

_s2t = OpenCC("s2twp")
_t2s = OpenCC("tw2sp")

# Abstract punctuation the model emits.
COMMA, PERIOD, QUESTION, EXCLAIM, ENUM = "<,>", "<.>", "<?>", "<!>", "<、>"
PUNCT = [COMMA, PERIOD, QUESTION, EXCLAIM, ENUM]
SPACE = "▁"
BLANK = "<b>"

_PUNCT_MAP = {
    "，": COMMA, ",": COMMA, "；": COMMA, ";": COMMA, "：": COMMA, ":": COMMA,
    "。": PERIOD, ".": PERIOD, "．": PERIOD, "｡": PERIOD,
    "？": QUESTION, "?": QUESTION,
    "！": EXCLAIM, "!": EXCLAIM,
    "、": ENUM, "､": ENUM,
}
# Kept inside words ("don't", "COVID-19", "3.5"); everything else that isn't a letter is dropped.
_KEEP_INNER = set("'-%&+")


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2FFFF
            or 0xF900 <= o <= 0xFAFF or 0x3040 <= o <= 0x30FF or o == 0x30FC or 0x31F0 <= o <= 0x31FF)


def is_hangul(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def normalize(text: str, lang: str, plain: bool = False) -> list[str]:
    """A transcript -> the symbol sequence the reader learns to emit. `plain`: a source written
    without punctuation or case (AISHELL, LibriSpeech…) — any stray marks are dropped and Latin
    is lower-cased, so the plain style never teaches the written style anything wrong."""
    if lang == "zh":
        text = _s2t.convert(text)
    # NFKC folds full-width Latin/digits to half-width; both widths of each mark map to one symbol.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("…", ".").replace("‘", "'").replace("’", "'")
    if plain:
        text = "".join(" " if c in _PUNCT_MAP else c for c in text.lower())
    out: list[str] = []
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in _PUNCT_MAP:
            p = _PUNCT_MAP[ch]
            # "3.5", "1,000": a period/comma between digits is part of the number.
            if ch in ".," and 0 < i < len(chars) - 1 and chars[i - 1].isdigit() and chars[i + 1].isdigit():
                out.append(ch)
                continue
            if out and out[-1] == SPACE:
                out.pop()
            if out and out[-1] in PUNCT:
                out[-1] = p  # "?!" -> one mark
            elif out:
                out.append(p)
        elif ch.isspace():
            if out and out[-1] != SPACE:
                out.append(SPACE)
        elif ch.isalnum():
            out.append(ch)
        elif ch in _KEEP_INNER and out and out[-1] not in (SPACE, *PUNCT):
            out.append(ch)
        # else: quotes, brackets, symbols -> dropped
    # Spaces only where they're written: between Latin/Hangul/digits, never next to CJK or punctuation.
    cleaned: list[str] = []
    for i, s in enumerate(out):
        if s == SPACE:
            prev = cleaned[-1] if cleaned else None
            nxt = out[i + 1] if i + 1 < len(out) else None
            if prev is None or nxt is None or prev in PUNCT or nxt in PUNCT:
                continue
            if (len(prev) == 1 and is_cjk(prev)) or (len(nxt) == 1 and is_cjk(nxt)):
                continue
        cleaned.append(s)
    return cleaned


def render(symbols: list[str], simplified: bool = False) -> str:
    """Symbols -> display text. A clause with any Han/kana in it ends in full-width punctuation
    (「講 machine learning。」), others in half-width plus a space; Latin next to Han/kana gets a
    space on each side, the way Taiwanese writing sets mixed text."""
    full = {COMMA: "，", PERIOD: "。", QUESTION: "？", EXCLAIM: "！", ENUM: "、"}
    half = {COMMA: ",", PERIOD: ".", QUESTION: "?", EXCLAIM: "!", ENUM: ","}
    # CTC can emit a mark twice with a blank between (". ."): one mark, the last one, is kept.
    merged: list[str] = []
    for s in symbols:
        if s in PUNCT and merged and merged[-1] in PUNCT:
            merged[-1] = s
        elif s == SPACE and merged and merged[-1] == SPACE:
            continue
        else:
            merged.append(s)
    symbols = merged
    out: list[str] = []
    clause_cjk = False
    for i, s in enumerate(symbols):
        if s == SPACE:
            if out and out[-1] != " ":
                out.append(" ")
        elif s in PUNCT:
            while out and out[-1] == " ":
                out.pop()
            out.append(full[s] if clause_cjk else half[s])
            if not clause_cjk and i + 1 < len(symbols):
                out.append(" ")
            clause_cjk = False
        else:
            cjk = is_cjk(s[0])
            prev = out[-1][-1] if out and out[-1] else ""
            if prev and prev != " " and prev not in "，。？！、" and (is_cjk(prev) != cjk) \
                    and (prev.isalnum() or is_cjk(prev)) and (s[0].isalnum() or cjk) and not is_hangul(s[0]):
                out.append(" ")
            clause_cjk = clause_cjk or cjk
            out.append(s)
    text = "".join(out).strip()
    if simplified:
        text = _t2s.convert(text)
    return text


class Vocab:
    def __init__(self, symbols: list[str]):
        self.itos = [BLANK] + [s for s in symbols if s != BLANK]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, syms: list[str]) -> list[int]:
        return [self.stoi[s] for s in syms if s in self.stoi]

    def encode_item(self, r) -> list[int]:
        return self.encode(r["sym"])

    def decode_ctc(self, ids) -> list[str]:
        out, prev = [], -1
        for i in ids:
            i = int(i)
            if i != prev and i != 0:
                out.append(self.itos[i])
            prev = i
        return out

    def save(self, path):
        json.dump(self.itos, open(path, "w"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        data = json.load(open(path))
        if isinstance(data, dict) and data.get("unit") == "gemma":
            from units import GemmaVocab
            return GemmaVocab.load(path)
        if isinstance(data, dict) and data.get("unit") == "hybrid":
            from units import HybridVocab
            return HybridVocab(data["itos"][1:])
        return cls(data)


def score_text(symbols: list[str]) -> str:
    """What accuracy is measured on: no punctuation, case-folded; CJK/Hangul per character, Latin per word.
    A mark stands for a word break ("bell, he" was written without a space beside the comma)."""
    s = "".join(" " if x == SPACE or x in PUNCT else x for x in symbols).lower()
    return " ".join(s.split())


if __name__ == "__main__":
    for t, l in [("这个软件的数据库，真的很好用！", "zh"), ("Hello, world. It's 3.5 GB!", "en"),
                 ("今日は「いい天気」ですね。", "ja"), ("안녕하세요, 반갑습니다.", "ko"),
                 ("我們今天要講 machine learning 的 gradient descent。", "zh")]:
        n = normalize(t, l)
        print(n, "->", render(n), "|", render(n, simplified=True))
