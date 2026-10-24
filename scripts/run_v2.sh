#!/bin/sh
# v2: hybrid units (characters for CJK/Hangul, Gemma tokens for Latin), all corpora minus the
# subtitle lines src/clean.py flagged. Dev = FLEURS validation + one held-out YODAS2 shard per language.
cd "$(dirname "$0")/.."
while pgrep -f "src/clean.py" >/dev/null || [ ! -f data/clean/drop.json ]; do sleep 30; done
TRAIN=$(ls feats/*.done | sed 's/\.done$//' | grep -v "_00000000$")
mkdir -p runs/v2
.venv/bin/python -u src/train.py \
  --train "feats/fleurs_*_train" $TRAIN \
  --dev "feats/fleurs_*_validation" feats/yodas2_zh000_00000000 feats/yodas2_ja000_00000000 feats/yodas2_ko000_00000000 feats/yodas2_en000_00000000 \
  --drop data/clean/drop.json --unit hybrid --up 2 \
  --out runs/v2 --layers 8 --steps 27500 --eval-every 2500 --batch-frames 12000 --lr 1e-3 --init runs/v2/step2500.pt --vocab runs/v2/vocab.json \
  > runs/v2/log.txt 2>&1
