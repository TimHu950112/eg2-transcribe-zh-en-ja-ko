#!/bin/sh
# Downloads the v4 model, the four n-gram packs (and optionally the reader checkpoint) from the
# GitHub release into models/. macOS gets Core ML, everything else LiteRT.
#   scripts/download_models.sh            # what transcribe.py needs
#   scripts/download_models.sh --litert   # LiteRT on macOS too (the model the Android app runs)
#   scripts/download_models.sh --reader   # also the PyTorch reader checkpoint, to train on from it
set -e
cd "$(dirname "$0")/.."
URL=https://github.com/TimHu950112/eg2-transcribe-zh-en-ja-ko/releases/download/v4.0
mkdir -p models
for f in transcribe_meta.json zh_hans.json lm_zh.bin lm_en.bin lm_ja.bin lm_ko.bin; do
  curl -fL --retry 5 -C - -o models/$f $URL/$f
done
if [ "$1" = "--reader" ]; then
  curl -fL --retry 5 -C - -o models/reader_v3.pt $URL/reader_v3.pt
  curl -fL --retry 5 -o models/vocab.json $URL/vocab.json
fi
if [ "$(uname)" = "Darwin" ]; then
  curl -fL --retry 5 -C - -o models/JoinTranscribe.mlpackage.zip $URL/JoinTranscribe.mlpackage.zip
  (cd models && unzip -oq JoinTranscribe.mlpackage.zip && rm JoinTranscribe.mlpackage.zip)
fi
if [ "$(uname)" != "Darwin" ] || [ "$1" = "--litert" ]; then
  curl -fL --retry 5 -C - -o models/join_transcribe.tflite $URL/join_transcribe.tflite
fi
du -sh models/*
