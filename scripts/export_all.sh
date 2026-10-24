#!/bin/sh
# Everything the apps download, in one folder: ./export_all.sh runs/v3/best.pt lm/mix '{"zh":[0.5,2.0],...}' export/v3
set -e
cd "$(dirname "$0")/.."
CK=$1; LMDIR=$2; W=$3; OUT=$4
mkdir -p $OUT
.venv/bin/python src/export_transcriber.py --ckpt $CK --out $OUT --target coreml --lm-weights "$W"
.venv-litert/bin/python src/export_transcriber.py --ckpt $CK --out $OUT --target litert --lm-weights "$W"
.venv/bin/python src/export_zh_hans.py $(dirname $CK)/vocab.json $OUT/zh_hans.json
for l in zh en ja ko; do cp $LMDIR/lm_$l.bin $OUT/lm_$l.bin; done
du -sh $OUT/*
