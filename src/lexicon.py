"""Known words for the beam's out-of-vocabulary cost (lm.ctc_beam `lex`): every lowercase Latin
word seen at least `--min` times in the transcripts, the YODAS2 subtitles and Wikipedia.

    python src/lexicon.py --lang en --out lm/big3/lex_en.txt
"""
import argparse, collections, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from lm import corpus
from lm_big import wiki_sentences
from text import SPACE, PUNCT

WORD = re.compile(r"^[a-z']+$")


def words(syms):
    cur = []
    for x in list(syms) + [SPACE]:
        if x == SPACE or x in PUNCT:
            if cur:
                yield "".join(cur).lower()
            cur = []
        else:
            cur.append(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--wiki-chars", type=int, default=50_000_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from text import normalize
    c = collections.Counter()
    for s in corpus(args.lang):
        c.update(words(s))
    for line in open(f"data/lmtext/{args.lang}.txt"):
        c.update(words(normalize(line.strip(), args.lang)))
    for s in wiki_sentences(args.lang, args.wiki_chars):
        c.update(words(s))
    keep = sorted(w for w, n in c.items() if n >= args.min and WORD.match(w))
    open(args.out, "w").write("\n".join(keep) + "\n")
    print(len(keep), "words", os.path.getsize(args.out) / 1e6, "MB")
