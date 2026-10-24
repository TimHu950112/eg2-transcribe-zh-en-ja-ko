# eg2-transcribe-zh-en-ja-ko

**離線語音轉文字，支援中文、English、日本語、한국어。** 模型完全跑在手機或電腦上，錄音不用上傳。

這是 [揪AI 逐字稿](#關於) 的 v4 模型。它以 Google [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) 的音訊編碼器（凍結、不改）為基礎，上面接一個自己訓練的 CTC「讀字器」（reader），再為每種語言配一個 n-gram 語言模型來挑字。

> English below ↓

## 起因

[john-rocky/eg2-transcribe](https://github.com/john-rocky/eg2-transcribe) 先證明了一件事：EmbeddingGemma 2 的音訊編碼器雖然是為了「語意嵌入」訓練的，但只要在上面接一個小小的讀字器，就能做出又快又準的**英文**語音辨識（LibriSpeech test-clean WER 4.2%）。

他的讀字器輸出的是 20 萬個英文單字，這對中文、日文行不通：中日文沒有空格可以切詞，詞表也永遠收不完人名和專有名詞。所以這個專案做了這些改動：

- **以字元為單位**：中、英、日、韓共用一份約 7,300 個符號的字表（漢字、假名、諺文、拉丁字母與標點），中英夾雜（「這個 function 會 return 一個 array」）也能自然寫出來。
- **四種語言一起訓練**：約 600 小時的公開語音（FLEURS、AISHELL-1、LibriSpeech、Zeroth-Korean、Common Voice、YODAS2），讀字器 3,040 萬參數，在一台 MacBook 上訓練。
- **每種語言一個 n-gram 語言模型**：讀字器聽得出音，但分不清同音字（覺得／決的）。beam search 搭配 n-gram（用逐字稿、YouTube 字幕和維基百科文字訓練）能少掉約 1/4 到 1/3 的錯誤。
- **標點與繁簡**：會自己加標點（中日文用全形，英韓用半形）；中文以台灣繁體輸出，也可轉成簡體。
- **可以上手機**：附 Core ML（iOS／macOS）和 LiteRT（Android）匯出檔，以及兩個平台的 beam search 解碼器原始碼。

## 準確度

錯誤率，越低越好。中文、日文、韓文是**字錯誤率（CER）**，英文是**詞錯誤率（WER）**。

| 語言 | 朗讀（FLEURS test） | 口語（YouTube 談話，YODAS2 保留集） | 不用語言模型（朗讀 / 口語） |
|---|---|---|---|
| 中文 | **18.7%** | **15.9%** | 25.6% / 18.0% |
| English | **24.3%** | **23.9%** | 38.4% / 36.9% |
| 日本語 | **18.2%** | **29.5%** | 23.2% / 32.6% |
| 한국어 | **9.8%** | **18.5%** | 13.7% / 22.6% |

- 換句話說，中文大約每 100 個字錯 16–19 個字、韓文朗讀錯不到 10 個，看得懂、適合做筆記和搜尋，但還不到可以直接當正式逐字稿的程度。
- 測試集完全沒有參與訓練或調參；語言模型的權重是在 FLEURS validation 上調的。
- 「口語」測試集的參考答案是 YouTube 上人工上的字幕，常常不是逐字照念（會被摘要或改寫），所以那一欄本身就有 10–30% 的誤差底線，看趨勢就好。
- **老實說**：在同一份測試集上，SenseVoice-Small 是中文 8.2% / 12.4%、英文 8.9% / 12.0%，比這個模型準很多。瓶頸在於 EmbeddingGemma 2 的編碼器是凍結的，它本來是為了理解語意而訓練，不是為了逐字拼寫。這個專案的優勢在於授權乾淨（Apache 2.0）、模型簡單、四種語言一份模型，而且整套訓練流程都公開，可以拿去改成你自己的語言。
- 英文 WER 比原作高，是因為測試集不同（FLEURS 是各種口音的維基百科句子，LibriSpeech 是有聲書），而且這裡是一個字母一個字母拼出來的，兩邊的數字不能直接比。

## 快速開始

需要 Python 3.10+ 和 [ffmpeg](https://ffmpeg.org/)。

```sh
git clone https://github.com/TimHu950112/eg2-transcribe-zh-en-ja-ko
cd eg2-transcribe-zh-en-ja-ko
pip install -r requirements.txt
scripts/download_models.sh            # 約 500 MB，放到 models/

python transcribe.py 錄音.m4a                    # 自動判斷語言
python transcribe.py 會議.mp3 --lang zh --hans   # 指定中文，輸出簡體
python transcribe.py 演講.wav --srt > 演講.srt   # 輸出字幕檔
```

在 macOS 上會用 Core ML（GPU），其他平台用 LiteRT（CPU）。也有網頁版測試工具：`python webtest/server.py`，然後打開 http://127.0.0.1:8770。

## 運作方式

```
16 kHz 音訊 ──► log-mel ──► EmbeddingGemma 2 音訊編碼器（凍結）──► 讀字器（8 層 Transformer, CTC）
               每 30 秒一個窗格、前後重疊 1 秒                      每 20 ms 前 8 名候選字
                                                                          │
     逐字稿（含時間、自動斷句） ◄── CTC beam search + 該語言的 n-gram ◄────┘
```

- 匯出的模型（`models/JoinTranscribe.mlpackage`、`models/join_transcribe.tflite`）輸入 30 秒音訊加語言代碼（0 = 自動，1–4 = zh/en/ja/ko），輸出每個 20 ms 的前 8 名符號和機率；梅爾頻譜在模型內計算。
- 解碼（beam search、語言判斷、斷句、繁轉簡）在模型外，`transcribe.py` 是 Python 版，`mobile/ios/TranscribeDecoder.swift` 和 `mobile/android/TranscribeDecoder.kt` 是手機版。
- n-gram 檔（`lm_*.bin`）是 6 階（英文 8 階）的 back-off 表，可以直接 memory-map。

| 檔案 | 大小 |
|---|---|
| `JoinTranscribe.mlpackage`（Core ML，8-bit） | 310 MB |
| `join_transcribe.tflite`（LiteRT，dynamic int8） | 313 MB |
| `lm_zh / en / ja / ko.bin` | 55 / 45 / 42 / 45 MB |

## 自己訓練

`src/` 是完整的訓練流程，`scripts/` 是實際跑 v4 用的指令。

1. `src/pipeline.py` 下載語料、切成句子，並用凍結的編碼器算出特徵存起來（`src/extract.py`）。編碼器不訓練，所以特徵只要算一次，之後讀字器的訓練非常快。
2. `src/clean.py` 用初版讀字器找出 YouTube 字幕裡「不是照念」的句子並剔除。
3. `src/train.py` 訓練讀字器（`scripts/run_v3.sh`）。
4. `src/yodas_text.py`、`src/lm_big.py` 建立 n-gram（`scripts/run_lms_big3.sh`）。
5. `src/eval_final.py`、`src/lm_grid.py` 調語言模型權重並評測。
6. `scripts/export_all.sh` 匯出 Core ML 和 LiteRT。

想加新語言，主要改 `src/text.py`（字表與正規化）和 `src/pipeline.py`（語料）。

## 授權

- 程式碼：Apache License 2.0（見 `LICENSE`）。
- 模型權重：EmbeddingGemma 2 為 Apache 2.0，使用時須遵守 [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)；本專案訓練的讀字器以 Apache 2.0 釋出。
- n-gram 語言模型含有從維基百科（CC BY-SA 4.0）與 YODAS2 字幕（CC BY 3.0）統計出來的資料，以 CC BY-SA 4.0 釋出。
- 訓練資料：FLEURS（CC BY 4.0）、AISHELL-1（Apache 2.0）、LibriSpeech（CC BY 4.0）、Zeroth-Korean（CC BY 4.0）、Common Voice（CC0）、YODAS2（CC BY 3.0）。

本專案與 Google 無關。

## 致謝

- [john-rocky/eg2-transcribe](https://github.com/john-rocky/eg2-transcribe)：證明了「凍結 EmbeddingGemma 2 編碼器 + 小讀字器」這條路可行，本專案從這裡出發。
- Google 的 EmbeddingGemma 2 團隊，以及上面列出的所有開放語料。

## 關於

這是 Join Note 筆記 App 裡「揪AI 逐字稿」功能的 v4 模型：錄完音，逐字稿直接在手機上產生。作者：胡廷鋒（Tim Hu）。

---

## English

**Offline speech-to-text for Chinese, English, Japanese and Korean**, running entirely on the phone or computer.

[john-rocky/eg2-transcribe](https://github.com/john-rocky/eg2-transcribe) showed that Google's EmbeddingGemma 2 audio encoder, kept frozen, plus a small trained CTC reader makes a fast and accurate **English** recognizer. Its reader outputs 200k English words, which doesn't carry over to Chinese or Japanese (no spaces, open vocabulary). This project:

- reads **characters**: one ~7,300-symbol set covering Han, kana, Hangul, Latin and punctuation, so code-switching ("這個 function 會 return 一個 array") just works;
- trains one 30.4M-parameter reader on **~600 h** of open speech in all four languages (FLEURS, AISHELL-1, LibriSpeech, Zeroth-Korean, Common Voice, YODAS2), on a MacBook;
- adds a per-language **n-gram + CTC beam search**, which removes a quarter to a third of the errors (homophones);
- writes punctuation, outputs Taiwan Traditional Chinese (optionally Simplified);
- ships **Core ML** and **LiteRT** exports plus Swift/Kotlin decoders.

### Accuracy

CER for zh/ja/ko, WER for en; lower is better. Test sets were never used for training or tuning.

| Language | Read (FLEURS test) | Spontaneous (YODAS2 held-out) | Without LM (read / spont.) |
|---|---|---|---|
| Chinese | **18.7%** | **15.9%** | 25.6% / 18.0% |
| English | **24.3%** | **23.9%** | 38.4% / 36.9% |
| Japanese | **18.2%** | **29.5%** | 23.2% / 32.6% |
| Korean | **9.8%** | **18.5%** | 13.7% / 22.6% |

The spontaneous set's references are YouTube subtitles that are often not verbatim, so it has a 10–30% error floor of its own. For honesty: SenseVoice-Small scores zh 8.2% / 12.4% and en 8.9% / 12.0% on the same items, so this model is not state of the art. The frozen encoder (trained for meaning, not spelling) is the ceiling. What this project offers is a clean Apache-2.0 stack, one small model for four languages, and a fully open training recipe you can adapt to your own language. English numbers aren't comparable with the original's LibriSpeech WER (different test set, letter-by-letter spelling).

### Quick start

```sh
pip install -r requirements.txt        # plus ffmpeg on PATH
scripts/download_models.sh             # ~500 MB into models/
python transcribe.py recording.m4a [--lang zh|en|ja|ko] [--hans] [--srt]
python webtest/server.py               # browser test page on http://127.0.0.1:8770
```

Training: see `src/` and `scripts/` (steps listed in the Chinese section above). Code is Apache-2.0; the n-gram packs are CC BY-SA 4.0 (built partly from Wikipedia); use of EmbeddingGemma 2 must follow the Gemma Prohibited Use Policy. Not affiliated with Google.
