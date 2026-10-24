#!/bin/sh
# v3: characters for every language (with the n-gram, letter-by-letter English beats whole Gemma
# tokens: 34.8% vs 41.2%), cleaned subtitles, continued from v1 (same unit set), full schedule.
cd "$(dirname "$0")/.."
TRAIN=$(ls feats/*.done | sed 's/\.done$//' | grep -v "_00000000$")
mkdir -p runs/v3
.venv/bin/python -u src/train.py \
  --train "feats/fleurs_*_train" $TRAIN \
  --dev "feats/fleurs_*_validation" feats/yodas2_zh000_00000000 feats/yodas2_ja000_00000000 feats/yodas2_ko000_00000000 feats/yodas2_en000_00000000 \
  --drop data/clean/drop.json --unit char --up 2 --vocab runs/v1/vocab.json --init runs/v1/step12500.pt \
  --out runs/v3 --layers 8 --steps 22000 --warmup 500 --eval-every 2500 --batch-frames 12000 --lr 5e-4 \
  > runs/v3/log.txt 2>&1
