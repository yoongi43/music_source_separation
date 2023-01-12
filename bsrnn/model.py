import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np


class BSRNN(nn.Module):
    def __init__(self, 
                 target_stem,                 
                 sr,
                 n_fft,
                 hop_length,
                 channels=2,
                 fc_dim=128,
                 group=16, # group of groupnorm in Band-Sequence modeling module
                 num_band_seq_module=12,
                 bands=[1000, 4000, 8000, 16000, 20000],  # Hz
                 num_subbands=[10, 12, 8, 8, 2, 1],
                 **kwargs):
        super().__init__()
        
        stem_list = ['mix', 'drums', 'bass', 'other', 'vocals']
        stem_dict = {stem:i for i,stem in enumerate(stem_list)}
        self.target_stem_idx = stem_dict[target_stem]
        ## params
        self.sr =sr
        self.n_fft = n_fft
        self.hop_length=hop_length
        self.channels=channels
        
        ### Modules
        self.bandsplit_module = BandSplitModule(sr=sr, n_fft=n_fft, channels=channels, fc_dim=fc_dim, bands=bands, num_subbands=num_subbands)
        self.bandseq_module = nn.Sequential(
            *[BandSeqModelingModule(channels=channels, fc_dim=fc_dim, group=group, num_subbands=num_subbands) for _ in range(num_band_seq_module)]
        )
        self.maskest_module = MaskEstimationModule(sr=sr, n_fft=n_fft, channels=channels, fc_dim=fc_dim, bands=bands, num_subbands=num_subbands)
        
    def forward(self, wav):
        ### Channels: mono or stereo
        # wav: (b, c, t), c=2
        ###
        wav = rearrange(wav, 'b c t -> (b c) t')  # INPUT
        
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, window=torch.hann_window(self.n_fft).to(wav.device), return_complex=False) # spec: ((b c), f, t, 2)
        spec_ = rearrange(spec, '(b c) f t ri -> b f t (c ri)', c=self.channels, ri=2)
        z = self.bandsplit_module(spec_)  
        q = self.bandseq_module(z)  # input: (b n k t) // out: (b n k t)
        mask = self.maskest_module(q)  # 실질적인 model output.
        
        cspec = torch.view_as_complex(spec)
        cmask = torch.view_as_complex(mask)
        est_cspec = cmask * cspec
        est_wav = torch.istft(est_cspec, n_fft=self.n_fft, hop_length=self.hop_length, window=torch.hann_window(self.n_fft).to(wav.device)) #((b c) t)
        est_spec = torch.view_as_real(est_cspec)  # ((b c) f t ri)
        est_wav = rearrange(est_wav, '(b c) t -> b c t', c=self.channels)
        est_spec = rearrange(est_spec, '(b c) f t ri -> b c f t ri', c=self.channels)
        return est_spec, est_wav, torch.abs(cmask)

"""
Modules
"""        
class BandSplitModule(nn.Module):
    def __init__(self, 
                 sr,
                 n_fft, 
                 channels=2, #stereo
                 fc_dim = 128,
                 bands=[1000, 4000, 8000, 16000, 20000],
                 num_subbands=[10, 12, 8, 8, 2, 1],
                ):
        # V7 is best
        # <1k: 100Hz bandwith / 1k~4k: 250Hz / 4k~8k: 500Hz / 8k~16k: 1kHz / 16k~20k: 2kHz / >20k: 1 subband. => 41 subbands. 
        super().__init__()
        self.bands = np.array(_quantize_hz2bins(sr, n_fft, bands, num_subbands))  # len(self.bands)==42
        assert len(self.bands)==sum(num_subbands)+1
        self.band_intervals = self.bands[1:]-self.bands[:-1]

        # self.ln_list = nn.ModuleList([nn.GroupNorm(1, band_interval) for band_interval in self.band_intervals])
        # self.fc_list = nn.ModuleList([nn.Linear(band_interval, fc_dim) for band_interval in self.band_intervals])
        self.layer_list = nn.ModuleList([
            nn.Sequential(
                # nn.GroupNorm(1, band_interval),  # LayerNorm
                ### Layernorm in F. not T.
                Rearrange('b f t c -> b t (f c)', c=2*channels), # 2: ri
                nn.LayerNorm(band_interval*channels*2), # with Rearrange('b f t c -> b t (f c)')
                # Rearrange('b f t c -> b t (f c)', c=2*channels),
                nn.Linear(2*channels*band_interval, channels*fc_dim),
                Rearrange('b t n -> b n 1 t')
                )
            for band_interval in self.band_intervals])
        
    def forward(self, cspec):
        # cspec: (b, f, t, (channel, ri))  # mono or stereo
        cspec_bands = [cspec[:, self.bands[i]:self.bands[i+1]] for i in range(len(self.bands)-1)]
        outs = []
        for cspec_band, layer in zip(cspec_bands, self.layer_list):
            out = layer(cspec_band)
            outs.append(out) # out:(b, n, 1, t)
        outs = torch.cat(outs, dim=-2) # outs: (b, n, k, t)
        return outs
    

class BandSeqModelingModule(nn.Module):
    def __init__(self,
                 fc_dim,
                 channels=2,
                 group=16,
                 num_subbands=[10, 12, 8, 8, 2, 1]):
        super().__init__()
        fc_dim = channels * fc_dim
        # group = channels * group
        # group = sum(num_subbands)
        hidden_dim = fc_dim  # BLSTM -> hiddendim = fc_dim 에서 2곱해짐
        num_total_subbands = sum(num_subbands)
        # self.blstm_seq = nn.Sequential(  # rnn across T
        #     Rearrange('b n k t -> (b k) n t'),
        #     nn.GroupNorm(num_groups=group, num_channels=fc_dim), # n/group
        #     Rearrange('(b k) n t -> (b k) t n', k=num_total_subbands), # (batch, seq, hidden)
        #     nn.LSTM(input_size=fc_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True), # out:(b, t, hidden_dim*2(bi))
        #     ExtractOutput(),
        #     nn.Linear(2*hidden_dim, fc_dim),
        #     Rearrange('(b k) t n -> b n k t', k=num_total_subbands)
        # )
        # self.blstm_band = nn.Sequential(  # rnn across K
        #     Rearrange('b n k t -> (b t) n k'),
        #     nn.GroupNorm(num_groups=group, num_channels=fc_dim),
        #     Rearrange('b_t n k -> b_t k n'),
        #     nn.LSTM(input_size=fc_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True),  #out: (b, k, hidden_dim*2)
        #     ExtractOutput(),
        #     nn.Linear(2*hidden_dim, fc_dim)
        # )
        self.blstm_seq = nn.Sequential(
            Rearrange('b n k t -> b k n t'),
            nn.GroupNorm(num_groups=num_total_subbands, num_channels=num_total_subbands),
            Rearrange('b k n t -> (b k) t n'),
            nn.LSTM(input_size=fc_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True),  #out: (b, k, hidden_dim*2)
            ExtractOutput(),
            nn.Linear(2*hidden_dim, fc_dim),
            Rearrange('(b k) t n -> b n k t', k=num_total_subbands)
        )
        self.blstm_band = nn.Sequential(
            Rearrange('b n k t -> b k n t'),
            nn.GroupNorm(num_groups=num_total_subbands, num_channels=num_total_subbands),
            Rearrange('b k n t -> (b t) k n'),
            nn.LSTM(input_size=fc_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True),  #out: (b, k, hidden_dim*2)
            ExtractOutput(),
            nn.Linear(2*hidden_dim, fc_dim)
        )
        
    def forward(self, z):
        # z: (b n k t)
        z_hat = self.blstm_seq(z) + z
        q = self.blstm_band(z_hat)
        q = rearrange(q, '(b t) k n -> b n k t', b=z_hat.size(0))
        q = q + z_hat
        return q
    

class MaskEstimationModule(nn.Module):
    def __init__(self, 
                 sr,
                 n_fft,
                 channels=2,
                 fc_dim = 128,
                 bands=[1000, 4000, 8000, 16000, 20000],
                 num_subbands=[10, 12, 8, 8, 2, 1],
                ):
        # V7 is best
        # <1k: 100Hz bandwith / 1k~4k: 250Hz / 4k~8k: 500Hz / 8k~16k: 1kHz / 16k~20k: 2kHz / >20k: 1 subband. => 41 subbands. 
        super().__init__()
        self.bands = np.array(_quantize_hz2bins(sr, n_fft, bands, num_subbands))  # len(self.bands)==42
        assert len(self.bands)==sum(num_subbands)+1
        self.band_intervals = self.bands[1:]-self.bands[:-1]
        num_total_subbands = len(self.bands)-1
        
        self.channels = channels
        fc_dim=fc_dim*channels
        hidden_dim = 4*fc_dim # *4
        
        ### For Parallel FC Layer, use convolution
        # self.pre_layer = nn.Sequential(  # for parallel processing
        #     Rearrange('b n k t -> b k n t'),
        #     nn.GroupNorm(num_total_subbands, num_total_subbands),  # LayerNorms
        #     nn.Conv2d(in_channels=num_total_subbands,
        #               out_channels=num_total_subbands * hidden_dim,
        #               kernel_size=(fc_dim, 1),
        #               groups=num_total_subbands), # first FC layers, out: (b, k*h, 1, t)
        #     Rearrange('b (k h) 1 t -> b k t h', h=hidden_dim)
        # )
        
        # self.post_fc_list = nn.ModuleList([  # second(last) fc layers of mlp
        #     nn.Linear(hidden_dim, band_interval*2*channels)
        #     for band_interval in self.band_intervals])
        
        
        ## Not parallel (for accurate LayerNorm)
        self.layer_list = nn.ModuleList([
            nn.Sequential(
                Rearrange('b n t -> b t n'),
                nn.LayerNorm(fc_dim),
                nn.Linear(fc_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, band_interval*2*channels),
                # Rearrange('b t n -> b n t')
            )
        for band_interval in self.band_intervals])
    
    def forward(self, q):
        # q: (b n k t)
        outs = []
        for i in range(len(self.band_intervals)):
            out = self.layer_list[i](q[:, :, i, :])
            out = rearrange(out, 'b t (f c ri) -> (b c) f t ri', c=self.channels, ri=2)
            outs.append(out)
        mask_real = torch.cat(outs, dim=-3)
        return mask_real
        
    # def forward(self, q):
    #     # For parallel processing
    #     # q: (b n k t)
    #     out1 = self.pre_layer(q) # out1:(b k t h)
    #     outs = []
    #     for i in range(len(self.band_intervals)):
    #         out2 = self.post_fc_list[i](out1[:,i]) # out2:(b, t, interval*2)
    #         out2 = rearrange(out2, 'b t (f c ri) -> (b c) f t ri', c=self.channels, ri=2)  #spec = rearrange(spec, '(b c) f t ri -> b f t (c ri)', c=self.hparams.channels, ri=2)
    #         outs.append(out2)
    #     mask_real = torch.cat(outs, dim=-3)
    #     return mask_real
            

class ExtractOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        out, _ = x
        return out


def _quantize_hz2bins(sr, n_fft, bands, num_subbands):
    assert len(bands)+1==len(num_subbands)
    bands = [0] + bands + [int(sr/2)]
    bands = np.array(bands)*n_fft/sr
    
    freq_bins = [] 
    for i in range(len(num_subbands)):
        start_freq = int(bands[i])
        end_freq = int(bands[i+1])
        num_bands = num_subbands[i]
        interval = (end_freq - start_freq) / num_bands
        for n in range(num_bands):
            freq_bins.append(int(start_freq+interval*n))
    freq_bins.append(int(n_fft/2)+1)
    return freq_bins

def adjust_seq_len(wav, hop_length):
    len_wav = wav.shape[-1]
    len_adj = int(len_wav/hop_length)*hop_length
    wav_adj = wav[...,:len_adj]
    return wav_adj



if __name__=="__main__":
    from torchinfo import summary
    sec = 6
    nfft=2048
    hop=512
    sr = 44100
    lr=None
    bs=None

    
    bsrnn = BSRNN('vocals', sr, nfft, hop, lr, bs, num_band_seq_module=3)
    wav_len = sr*sec
    wav_len = int(wav_len/hop)*hop
    summary(bsrnn, input_size=(2, 2, wav_len))
    
    
    # wav = torch.randn(4, 2, sr*sec)
    # print('wavlen: ', int(wav.shape[-1]/hop)*(hop))
    # wav = wav[:, :int(wav.shape[-1]/hop)*(hop)]
    # print('in wav: ', wav.shape)
    
    # out_spec, out_wav = bsrnn.forward(wav)
    # print('out wav:', out_wav.shape)
    # print('out spec:', out_spec.shape)