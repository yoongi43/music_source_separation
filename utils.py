import torch
import demucs.demucs.augment as augment
from einops import rearrange
import numpy as np
from metrics import cmgan_snr, museval_sdr, new_sdr
from time import time
import librosa 
import librosa.display
import matplotlib.pyplot as plt
# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__
    
class DotConfig:
    def __init__(self, cfg):
        self._cfg = cfg
    def __getattr__(self, k):
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v
    

def augment_modules(args):
    # shift=44100*3
    # augments = [augment.Shift(shift=int(args.dset.samplerate * args.dset.shift),
    #                               same=args.augment.shift_same)]
    augments = []
    if args.augment.flip:
        augments += [augment.FlipChannels(), augment.FlipSign()]
    for aug in ['scale', 'remix']:  #! 야매로 짰음. dictionary dot access문제 관련
        # kw = getattr(args.augment, aug)
        # if kw.proba:
            # augments.append(getattr(augment, aug.capitalize())(**kw))
            
        if aug=='scale':
            kw = args.augment.scale
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(proba=kw.proba, min=kw.min, max=kw.min))
                
        elif aug=='remix':
            kw = args.augment.remix
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(proba=kw.proba, group_size=kw.group_size))
    
    return torch.nn.Sequential(*augments)
        
        
def adjust_seq_len(wav, hop_length):
    len_wav = wav.shape[-1]
    len_adj = int(len_wav/hop_length)*hop_length
    wav_adj = wav[...,:len_adj]
    return wav_adj


def apply_model_overlap_add(model,
                            mix,
                            sr=44100,
                            hop_length=512, # model's hop size
                            shifts=1.5,
                            segments=3,
                            normalize=True
                            ):
    # apply model with overlap-add
    # mix: (b, c, t)
    model.eval()
    shifts = int(shifts*sr)
    segments = int(segments*sr)
    segments = int(segments / hop_length) * hop_length
    overlap = segments - shifts
    
    mix_len = mix.shape[-1]
    idx = 0
    est_wav = torch.zeros_like(mix).to(mix.device)
    while True:
        start = idx*shifts
        # print(start/44100)
        end = idx*shifts + segments
        
        if end > mix_len:  # Last segment
            mix_seg = mix[:, :, start:]
            mix_seg = adjust_seq_len(mix_seg, hop_length=hop_length)
            mix_seg, gain = rms_normalize_torch(mix_seg, ref_dBFS=-10, return_gain=True)
            _, est_seg, _ = model(mix_seg)
    
            len_seg = mix_seg.shape[-1]  # ==est_seg.shape[-1]
            if start + overlap > mix_len:
                est_wav[..., start:start+len_seg] = (est_wav[..., start:] + est_seg)/2
            else:
                est_wav[..., start:start+overlap] = (est_wav[..., start:start+overlap] + est_seg[..., :overlap])/2
                est_wav[..., start+overlap:start+len_seg]=est_seg[..., overlap:] ## 전제조건: segment는 overlap보다 훨씬 클 것이다. (그 범위가 hop size보다 커야함. )
                
            return est_wav[..., :start+len_seg]
        
        mix_seg = mix[:, :, start:end]
        mix_seg, gain = rms_normalize_torch(mix_seg, ref_dBFS=-10, return_gain=True)
        est_spec, est_seg, _ = model(mix_seg)
        if idx==0:
            est_wav[:, :, start:end] = est_seg
        else:
            est_wav[:, :, start:start+overlap] = (est_wav[:, :, start:start+overlap] + est_seg[:, :, :overlap]) / 2
            est_wav[:, :, start+overlap:end] = est_seg[:, :, overlap:]
        idx += 1
        
    # return est_wav[..., :start+len_seg]
    
    
def cal_metrics(ref, est, sr, chunks=1):
    # ref, est:(b c t)
    target_wav = rearrange(ref, 'b c t -> b t c')
    est_wav = rearrange(est, 'b c t -> b t c')
    """ cSDR"""
    st = time()
    # print(target_wav.shape)  # nn.DataParallel 쓰면 다 모여서 계산됨. 
    csdr = museval_sdr(ref=target_wav, est=est_wav, sr=sr, chunks=chunks) # batch=12일때 대략 90초 걸림
    # csdr = np.median(csdr, axis=-1)
    # csdr = np.nan_to_num(csdr, nan=0.0)
    # print('cSDR time:' , time()-st)
    """ uSDR==SNR"""
    st = time()
    usdr = cmgan_snr(clean_speech=target_wav, processed_speech=est_wav)
    usdr = np.nan_to_num(usdr, nan=0.0)
    # print('SNR time: ', time()-st)
    
    # """ new SDR"""
    # nsdr = new_sdr(references=target_wav, estimates=est)
    # nsdr = torch.nan_to_num(nsdr)
    # print('nsdr time:', nsdr)
    # nsdr = new_sdr(references=target_wav, estimates=est_wav)
    return{'csdr':csdr, 'usdr':usdr}
    
def get_numpy(x):
    if type(x)==torch.Tensor:
        x = x.detach().cpu().numpy()
    return x

def get_spectrogram(clean, pred, mix, mask, sr, batch_idx, n_fft=2048, hop_length=512):
    def get_data(x):
        if batch_idx is None:
            return get_numpy(x)
        else:
            return get_numpy(x)[batch_idx]
    def get_spec(x):
        return 20*np.log10(np.abs(librosa.stft(x,n_fft=n_fft,hop_length=hop_length))+1e-7)
    def plot_spec(x, ax, mode='spec'):
        if   mode == 'diff': 
            cmap, vmin, vmax = 'bwr', -20, 20
        elif mode == 'spec': 
            cmap, vmin, vmax = 'jet', -40, 40
        elif mode == 'mask':
            cmap, vmin, vmax = 'RdGy_r', -40, 40
        return librosa.display.specshow(x, hop_length=100, x_axis='time', y_axis='hz', 
                                        sr=sr, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)
        
        
    clean, pred, mix, mask = get_data(clean), get_data(pred), get_data(mix), get_data(mask)
    clean, pred, mix = clean.flatten(), pred.flatten(), mix.flatten()

    """ normalize"""
    # noisy, gain = rms_normalize_0db(noisy)
    # pred, clean = pred*gain, clean*gain
    
    # 만약 위에걸 안썼다면...
    gain = 1

    clean_spec, pred_spec, mix_spec = get_spec(clean), get_spec(pred), get_spec(mix)
    fig, ax = plt.subplots(2, 2)
    img = plot_spec(clean_spec, ax[0, 0]); fig.colorbar(img, ax=ax[0, 0], format='%+2.f')
    img = plot_spec(pred_spec, ax[0, 1]); fig.colorbar(img, ax=ax[0, 1], format='%+2.f')
    img = plot_spec(mix_spec, ax[1, 0]); fig.colorbar(img, ax=ax[1, 0], format='%+2.f')
    img = plot_spec(20*np.log10(mask**(10/3)+1e-10), ax[1, 1], mode='mask') ; fig.colorbar(img, ax=ax[1, 0], format='%+2.f')
    ax[0, 0].set_title('LMS(gt)')
    ax[0, 1].set_title('LMS(pred)')
    ax[1, 0].set_title('LMS(mix)')
    ax[1, 1].set_title('log mask')
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')
            ax[i, j].set_yticks([])
    
    fig.set_size_inches(10, 12)
    fig.subplots_adjust(wspace=0, hspace=0.1)
    return fig


def rms_normalize(x, ref_dBFS=-23.0, eps=1e-12, return_gain=False):
    def cal_rms(x): return np.sqrt(np.mean(np.square(x), axis=-1)+eps)
    rms = cal_rms(x)
    ref_linear = np.power(10, (ref_dBFS-3.0103)/20.)
    gain = ref_linear/(rms+eps)
    return x*gain, gain if return_gain else x*gain

def rms_normalize_torch(x, ref_dBFS=-23.0, eps=1e-12, return_gain=False):
    def cal_rms(x): return torch.sqrt(torch.mean(torch.square(x), axis=-1)+eps)
    rms = cal_rms(x)
    # ref_linear = torch.pow(10, (ref_dBFS-3.0103)/20.)
    ref_linear = 10**((ref_dBFS-3.0103)/20.)
    gain = ref_linear / (rms+eps)
    gain = gain.view(-1, 1, 1)
    
    return x*gain, gain if return_gain else x*gain

      
                        
def linear2db(x, eps=1e-5, scale=20):
    return scale * np.log10(x + eps)


def db2linear(x, eps=1e-5, scale=20):
    return 10 ** (x / scale) - eps