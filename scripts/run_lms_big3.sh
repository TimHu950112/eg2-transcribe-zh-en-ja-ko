#!/bin/sh
# n-grams with spoken text: YODAS2 subtitles of every shard (src/yodas_text.py) next to the
# transcripts and Wikipedia.
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u src/lm_big.py"
$PY --lang ko --order 6 --wiki-chars 0 --transcript-weight 4 --talk-chars 60000000 --budget 5000000 --out lm/big3/lm_ko.bin > logs/lm3_ko.txt 2>&1 &
$PY --lang ja --order 6 --wiki-chars 10000000 --transcript-weight 4 --talk-chars 40000000 --budget 5000000 --out lm/big3/lm_ja.bin > logs/lm3_ja.txt 2>&1 &
wait
$PY --lang zh --order 6 --wiki-chars 60000000 --transcript-weight 8 --talk-chars 20000000 --talk-weight 4 --budget 6000000 --out lm/big3/lm_zh.bin > logs/lm3_zh.txt 2>&1 &
while ! grep -q en000 logs/yodas_text.txt; do sleep 10; done
$PY --lang en --order 8 --wiki-chars 50000000 --transcript-weight 8 --talk-chars 60000000 --budget 6000000 --out lm/big3/lm_en.bin > logs/lm3_en.txt 2>&1 &
wait
echo DONE > logs/lm3_done.txt
