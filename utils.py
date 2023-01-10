import torch
import demucs.demucs.augment as augment
from einops import rearrange
import numpy as np
from metrics import cmgan_snr, museval_sdr, new_sdr
from time import time

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
            _, est_seg = model(mix_seg)
    
            len_seg = mix_seg.shape[-1]  # ==est_seg.shape[-1]
            if start + overlap > mix_len:
                est_wav[..., start:start+len_seg] = (est_wav[..., start:] + est_seg)/2
            else:
                est_wav[..., start:start+overlap] = (est_wav[..., start:start+overlap] + est_seg[..., :overlap])/2
                est_wav[..., start+overlap:start+len_seg]=est_seg[..., overlap:] ## 전제조건: segment는 overlap보다 훨씬 클 것이다. (그 범위가 hop size보다 커야함. )
                
            return est_wav[..., :start+len_seg]
        
        mix_seg = mix[:, :, start:end]
        est_spec, est_seg = model(mix_seg)
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
    
                        