import io
import numpy as np
import soundfile as sf
import librosa


def load_audio(src, sr=16000):
    """A path (any format soundfile/librosa reads) -> mono float32 at `sr`."""
    try:
        wav, r = sf.read(src, dtype="float32", always_2d=True)
        wav = wav.mean(1)
    except Exception:
        wav, r = librosa.load(src, sr=None, mono=True)
    if r != sr:
        wav = librosa.resample(wav, orig_sr=r, target_sr=sr)
    return np.ascontiguousarray(wav, dtype=np.float32)
