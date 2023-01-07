import torch as th
import torch
import numpy as np
import math
import museval



"""
SNR
SDR
uSDR
cSDR
"""

EPS = 1e-10

def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    # assert references.dim() == 4 #?? batch, channel, length,......
    # assert estimates.dim() == 4
    assert references.dim()==3
    assert estimates.dim()==3
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(-2, -1))
    den = th.sum(th.square(references - estimates), dim=(-2, -1))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores  # 그냥 SNR 같은디...


def museval_sdr(ref, est, sr=44100, chunks=1):
    clean = get_numpy(ref) + EPS
    est = get_numpy(est) + EPS
    if chunks is None:
        win = clean.shape[-1]
        hop = clean.shape[-1]
    else:
        win = sr*chunks
        hop = sr*chunks
    batch_size = clean.shape[0]
    assert batch_size < 1026 # 
    # sdr, _, _, _ = museval.evaluate(references=clean, estimates=est, win=win, hop=hop, padding=True)
    sdrs = []
    for b in range(batch_size):  # 이게 더 빠름
        sdr, _, _, _ = museval.evaluate(references=clean[b][None], estimates=est[b][None], win=win, hop=hop, padding=True)
        sdrs.append(sdr)
        
    # msdr, i, j, k = museval.evaluate(references=a, estimates=n, win=1*sr, hop=1*sr, padding=True)
    return sdrs

def cmgan_snr(clean_speech, processed_speech, sample_rate=None):
    # Check the length of the clean and processed speech. Must be the same.
    # clean_length = len(clean_speech)
    # processed_length = len(processed_speech)
    clean_speech, processed_speech = get_numpy(clean_speech), get_numpy(processed_speech)
    clean_length = clean_speech.shape[-2]
    processed_length = processed_speech.shape[-2]
    if clean_length != processed_length:
        raise ValueError('Both Speech Files must be same length.')
    overall_snr = 10 * np.log10(np.sum(np.square(clean_speech), axis=(-2, -1))+1e-10 / (np.sum(np.square(clean_speech - processed_speech), axis=(-2, -1))+1e-10))

    # ## Global Variables
    # winlength = round(30 * sample_rate / 1000)    # window length in samples
    # skiprate = math.floor(winlength / 4)     # window skip in samples
    # MIN_SNR = -10    # minimum SNR in dB
    # MAX_SNR = 35     # maximum SNR in dB

    # ## For each frame of input speech, calculate the Segmental SNR
    # num_frames = int(clean_length / skiprate - (winlength / skiprate))   # number of frames
    # start = 0      # starting sample
    # window = 0.5 * (1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)))

    # segmental_snr = np.empty(num_frames)
    # EPS = np.spacing(1)
    # for frame_count in range(num_frames):
    #     # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
    #     clean_frame = clean_speech[start:start + winlength]
    #     processed_frame = processed_speech[start:start + winlength]
    #     clean_frame = np.multiply(clean_frame, window)
    #     processed_frame = np.multiply(processed_frame, window)

    #     # (2) Compute the Segmental SNR
    #     signal_energy = np.sum(np.square(clean_frame))
    #     noise_energy = np.sum(np.square(clean_frame - processed_frame))
    #     segmental_snr[frame_count] = 10 * math.log10(signal_energy / (noise_energy + EPS) + EPS)
    #     segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
    #     segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)

    #     start = start + skiprate

    # return overall_snr, segmental_snr

    return overall_snr


def get_numpy(x):
    if type(x) == torch.Tensor:
        x = x.detach().cpu().numpy()
    return x



if __name__=="__main__":
    import soundfile as sf
    batch=2
    sr=44100
    sec=6
    samp = sr*sec
    c=2
    # b = a
    # a = torch.randn(batch, samp, c)
    # b = torch.randn(b, samp, c)
    start = sec*30
    
    a, _ = sf.read('./temp/1.wav')
    b, _ = sf.read('./temp/0.wav')

    a = np.tile(a[start:start+samp], (batch, 1, 1))
    b = np.tile(b[start:start+samp], (batch, 1, 1))
    n = np.random.randn(*a.shape)*3
    n = n+a
    print(a.shape)
    print(b.shape)
    
    # new_sdr, museval_sdr, snr
    # nsdr = new_sdr(references=a, estimates=b)
    msdr = museval_sdr(ref=a, est=n, sr=sr, chunks=1)
    print('museval sdr: ', msdr)
    cmsnr = cmgan_snr(clean_speech=a, processed_speech=n, sample_rate=sr)
    # print('new sdr: ', nsdr)
    print('cmgan snr: ', cmsnr)