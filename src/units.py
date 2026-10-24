"""Gemma's own tokens as the reader's output units (instead of single characters).

EmbeddingGemma 2's audio encoder was trained to hand Gemma soft tokens, and its frames sit
near Gemma token embeddings (" singing", "没有", " 日本", "는"). Writing those tokens instead of
letters spares the reader spelling English letter by letter ("polise", "aday") and lets a
Chinese word come out whole (覺得, not 決的).

The units are the Gemma tokens the training transcripts use (Traditional Chinese with Taiwan
wording, as `render` writes it); `decode_ctc` gives back the same symbols the character
reader produces, so both are scored the same way.
"""
import collections, json

from transformers import AutoTokenizer

from text import normalize, render

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
    return _tok


class GemmaVocab:
    unit = "gemma"

    def __init__(self, gemma_ids):
        # 0 = blank, 1 = a token outside the kept set.
        self.gids = [-1, -2] + [g for g in gemma_ids]
        self.index = {g: i for i, g in enumerate(self.gids) if g >= 0}

    def __len__(self):
        return len(self.gids)

    @staticmethod
    def text_of(r):
        return render(r["sym"])

    def encode_item(self, r):
        if "tgt" not in r:
            r["tgt"] = [self.index.get(g, 1) for g in tokenizer().encode(self.text_of(r), add_special_tokens=False)]
        return r["tgt"]

    def decode_ctc(self, ids):
        out, prev = [], -1
        for i in ids:
            i = int(i)
            if i != prev and i > 1:
                out.append(self.gids[i])
            prev = i
        return normalize(tokenizer().decode(out), "en")

    @classmethod
    def build(cls, items, min_count=2):
        c = collections.Counter()
        tok = tokenizer()
        for r in items:
            c.update(tok.encode(cls.text_of(r), add_special_tokens=False))
        return cls([g for g, n in c.most_common() if n >= min_count])

    def save(self, path):
        json.dump({"unit": "gemma", "ids": self.gids[2:]}, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path))["ids"])


from text import Vocab, SPACE, PUNCT

_INNER = set("'-%&+.")


def _latin(s):
    return len(s) == 1 and s.isascii() and s.isalnum()


class HybridVocab(Vocab):
    """Characters for 中文 / 日本語 / 한국어, Gemma tokens for Latin runs ("§<id>").

    The unit experiment showed each kind of writing prefers its own unit: Gemma tokens cut the
    English error by a third (no more letter-by-letter spelling), while Han / kana / Hangul did
    better one character at a time (a few thousand characters are learnt far faster than tens
    of thousands of multi-character tokens). A Latin run inside a Chinese sentence
    ("講 gradient descent 的") is tokenized the same way as English.
    """

    unit = "hybrid"

    @staticmethod
    def to_units(sym):
        out, i, n = [], 0, len(sym)
        tok = tokenizer()
        while i < n:
            if _latin(sym[i]):
                j = i + 1
                while j < n and (_latin(sym[j]) or (sym[j] in _INNER) or
                                 (sym[j] == SPACE and j + 1 < n and _latin(sym[j + 1]))):
                    j += 1
                text = "".join(" " if x == SPACE else x for x in sym[i:j])
                out += ["§%d" % g for g in tok.encode(text, add_special_tokens=False)]
                i = j
            else:
                out.append(sym[i]); i += 1
        return out

    @staticmethod
    def from_units(units):
        out, run = [], []

        def flush():
            if run:
                for c in tokenizer().decode(run):
                    out.append(SPACE if c == " " else c)
                run.clear()
        for u in units:
            if u.startswith("§"):
                run.append(int(u[1:]))
            else:
                flush(); out.append(u)
        flush()
        return out

    def encode_item(self, r):
        if "tgt" not in r:
            r["tgt"] = self.encode(self.to_units(r["sym"]))
        return r["tgt"]

    def decode_ctc(self, ids):
        return self.from_units(super().decode_ctc(ids))

    @classmethod
    def build(cls, items, min_token_count=2):
        c = collections.Counter()
        for r in items:
            c.update(cls.to_units(r["sym"]))
        return cls([u for u, k in c.most_common() if not u.startswith("§") or k >= min_token_count])

    def save(self, path):
        json.dump({"unit": "hybrid", "itos": self.itos}, open(path, "w"), ensure_ascii=False)
