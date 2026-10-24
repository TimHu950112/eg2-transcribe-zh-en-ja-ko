#!/bin/sh
# Character vs Gemma-token output units, same ~150 h subset, same steps.
cd "$(dirname "$0")/.."
SUB="feats/fleurs_*_train feats/yodas2_*000_0000000[123] feats/aishell_000 feats/aishell_020 feats/zeroth_train-00000-of-00006-6f62b6637bde6a2a feats/zeroth_train-00001-of-00006-a6254d939f218ad2 feats/libri100_0 feats/libri100_1 feats/libri100_2 feats/cv_ja_train_0 feats/cv_ja_train_1"
DEV="feats/fleurs_*_validation feats/yodas2_zh000_00000000 feats/yodas2_ja000_00000000 feats/yodas2_ko000_00000000 feats/yodas2_en000_00000000"
for arm in ${ARMS:-char gemma}; do
  mkdir -p runs/unit_$arm
  case $arm in char) EXTRA="--unit char --up 2";; gemma) EXTRA="--unit gemma --up 1 --head-from-gemma";; hybrid) EXTRA="--unit hybrid --up 2 --head-from-gemma";; hybrid_big) EXTRA="--unit hybrid --up 2 --head-from-gemma --layers 12 --d 768 --heads 12 --ff 3072 --lr 7e-4";; esac
  .venv/bin/python -u src/train.py --train $SUB --dev $DEV --out runs/unit_$arm --layers 6 --steps 3000 --eval-every 1500 --batch-frames 12000 $EXTRA > runs/unit_$arm/log.txt 2>&1
done
