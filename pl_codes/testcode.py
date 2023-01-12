import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio

def linear2db(x, eps=1e-5, scale=20):
    return scale * np.log10(x + eps)


def db2linear(x, eps=1e-5, scale=20):
    return 10 ** (x / scale) - eps

sr = 44100
s = 6
meter = pyln.Meter(sr)

target_lufs = -14.
wav = np.random.randn(sr*s)
loud = meter.integrated_loudness(wav)


wav_norm = pyln.normalize.loudness(wav, loud, target_lufs)
loud_norm = meter.integrated_loudness(wav_norm)

gain = target_lufs - loud
wav_gain = wav * db2linear(gain)
loud_gain = meter.integrated_loudness(wav_gain)

print('before loud:', loud)
print('after loud: ', loud_norm)
print("after gain: ", loud_gain)

### torch
meter = torchaudio.transforms.Loudness(sr)
wav = torch.randn(sr*s)
