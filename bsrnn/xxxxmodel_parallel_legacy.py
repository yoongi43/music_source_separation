import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
# from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule, LightningModule, Trainer


class BSRNN(LightningModule):
# class BSRNN(nn.Module):
    def __init__(self, 
                 sr,
                 n_fft,
                 hop_length,
                 lr,
                 batch_size,
                 num_band_seq_module=12,
                 **kwargs):
        super().__init__()
        # self.save_hyperparameters()  # pytorch_lightning
        self.sr=sr
        self.n_fft=n_fft
        self.hop_length=hop_length
        ## params
        fc_dim = 128
        bands=[1000, 4000, 8000, 16000, 20000]
        num_subbands=[10, 12, 8, 8, 2, 1]
        group=16
        ## params
        
        self.bandsplit_module = BandSplitModule(sr=sr, fc_dim=fc_dim, bands=bands, num_subbands=num_subbands)
        self.bandseq_modeling_module = nn.Sequential(
            *[BandSeqModelingModule(fc_dim=fc_dim, group=group) for _ in range(num_band_seq_module)]
        )
        self.maskest_module = MaskEstimationModule(sr=sr, fc_dim=fc_dim, bands=bands, num_subbands=num_subbands)
    
    def forward(self, wav):
        stft = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, window=torch.hann_window(self.n_fft).to(wav.device), return_complex=False)
        # stft = (b, f, t, 2)
        z = self.bandsplit_module(stft)
        q = self.bandseq_modeling_module(z)
        mask = self.maskest_module(q)
        stft = torch.view_as_complex(stft)
        mask = torch.view_as_complex(mask)
        est_stft = mask * stft
        est_wav = torch.istft(est_stft, n_fft=self.n_fft, hop_length=self.hop_length, window=torch.hann_window(wav.size(-1)).to(wav.device))
        est_stft = torch.view_as_real(est_stft)
        return est_stft, est_wav
    
    def training_step(self, batch, batch_idx):
        pass
        

class BandSplitModule(nn.Module):
    def __init__(self, 
                 sr,
                 bands=[1000, 4000, 8000, 16000, 20000],
                 num_subbands=[10, 12, 8, 8, 2, 1],
                 fc_dim = 128):
        # V7 is best
        # <1k: 100Hz bandwith / 1k~4k: 250Hz / 4k~8k: 500Hz / 8k~16k: 1kHz / 16k~20k: 2kHz / >20k: 1 subband. => 41 subbands. 
        super().__init__()
        self.fs = int(sr/2)
        self.fc_dim = fc_dim
        assert len(bands)+1==len(num_subbands)
        self.bands = [0] + bands + [self.fs]  # [0, 1000, 4000, 8000, 16000, 20000, 22050]
        self.num_subbands = num_subbands      # [10,  12,    8,    8,     2,     1]
        
        self.bands_ranges = [(self.bands[i+1]-self.bands[i]) for i in range(len(num_subbands))]  # [1000, 3000, 4000, 8000, 4000, 2050]
        self.width_subbands = [self.bands_ranges[i] // num_subbands[i] for i in range(len(num_subbands))] 
        # [100, 250, 500, 1000, 2000, rest]
        
        self.ln_list = nn.ModuleList([nn.GroupNorm(self.num_subbands[i], self.bands_ranges[i]) for i in range(len(self.num_subbands))]) # LayerNorm: For all dimension in a batch
        self.fc_list = nn.ModuleList([nn.Conv2d(
            in_channels=self.bands_ranges[i], # self.width_subbands[i] * num_subbands[i]
            out_channels=fc_dim * num_subbands[i],
            kernel_size=(1, 2),
            groups=self.num_subbands[i]
            ) for i in range(len(num_subbands))])
        # self.fc_list = nn.ModuleList([nn.Linear(2*bw, fc_dim) for bw in self.width_subbands])  # bw:complex -> 2*bw:concat[real,imag]
        # self.ln_list = nn.ModuleList([nn.LayerNorm(2*bw) for bw in self.width_subbands])  #! Freq dimension normalization...?
        
    def forward(self, cspec):
        # cspec: (b, f, t, 2)
        cspec_bands = [cspec[:, self.bands[i]:self.bands[i+1]] for i in range(len(self.num_subbands))]
        # outs = torch.empty((cspec_bands[0].size(0), self.fc_dim, 0, cspec_bands[0].size(-2)), requires_grad=True)
        outs = []
        for cspec_band, ln, fc, num_subband in zip(cspec_bands, self.ln_list, self.fc_list, self.num_subbands):
            # spec = rearrange(cspec_band, 'b f t c -> b t (f c)', c=2)  # last layer == 2*bw
            # out = fc(ln(spec)) # out: (b t N) (N=fc_dim)
            
            # cspec_band: (b f t 2). ex) f: 1000~4000Hz. split: 12
            cspec_band = ln(cspec_band)
            out = fc(cspec_band)  # -> (b, n*ks, t, 1)
            out = rearrange(out, 'b (n s) t 1 -> b n s t', s=num_subband)
            outs.append(out)
        outs = torch.cat(outs, dim=-1)  # (b, n, k, t)
        return outs # (b N K T)
    

class BandSeqModelingModule(nn.Module):
    def __init__(self,
                 batch_size=4,
                 fc_dim = 128,
                 group=16,  # 2, 4, 8, 16...128
                 num_subbands=[10, 12, 8, 8, 2, 1]
                #  hidden_dim = 256
                 ):
        super().__init__()
        # k=41, n = fc_dim=128
        hidden_dim = 2 * fc_dim
        num_total_subbands = sum(num_subbands)
        self.blstm_seq = nn.Sequential(  # rnn across T
            Rearrange('b n k t -> (b k) n t'),
            nn.GroupNorm(num_groups=group, num_channels=fc_dim),  # n/group
            Rearrange('(b k) n t -> (b k) t n', k=num_total_subbands), # (batch, seq, hidden)
            nn.LSTM(input_size=fc_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True), # out:(b, t, hidden_dim*2(bi))
            nn.Linear(hidden_dim, fc_dim),
            Rearrange('(b k) t n -> b n k t', k=num_total_subbands)
        )
        self.blstm_band = nn.Sequential(  # rnn across K
            Rearrange('b n k t -> (b t) n k'),
            nn.GroupNorm(num_groups=group, num_channels=fc_dim),
            Rearrange('b_t n k -> b_t k n'),
            nn.LSTM(input_size=fc_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True), # out: (b, k, hidden_dim*2)
            nn.Linear(hidden_dim, fc_dim),
            # Rearrange('(b t) k n -> b n k t')
        )
        
    
    def forward(self, z):
        # z: (b, n, k, t)
        ## sequence-level rnn: across T
        ## K subband features share a same RNN since they have the same feature dimension N.
        z_hat = self.blstm_band(z) + z
        ## band-level rnn: across K
        q = self.blstm_band(z_hat)
        q = rearrange(q, '(b t) k n -> b n k t', b=z_hat.size(0))
        q = q + z_hat
        
        return q
        
        
class MaskEstimationModule(nn.Module):
    def __init__(self,
                 sr,
                 fc_dim,
                 bands=[1000, 4000, 8000, 16000, 20000],
                 num_subbands=[10, 12, 8, 8, 2, 1]                 
                 ):
        super().__init__()
        self.fs = int(sr/2)
        self.fc_dim = fc_dim
        assert len(bands)+1==len(num_subbands)
        self.bands = [0] + bands + [self.fs]  # [0, 1000, 4000, 8000, 16000, 20000, 22050]
        self.num_subbands = num_subbands      # [10,  12,    8,    8,     2,     1]
        self.num_subbands_cum = np.cumsum([0]+num_subbands)
        
        self.bands_ranges = [(self.bands[i+1]-self.bands[i]) for i in range(len(num_subbands))]  # [1000, 3000, 4000, 8000, 4000, 2050]
        self.width_subbands = [self.bands_ranges[i] // num_subbands[i] for i in range(len(num_subbands))]  # [100, 250, 500, 1000, 2000, rest]
        
        
        hidden_dim = 4*fc_dim
        self.ln_list = nn.ModuleList([nn.GroupNorm(ns, ns) for ns in num_subbands])
        self.mlp_list = nn.ModuleList([   # input: (b k n t)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=num_subbands[i],
                    out_channels=hidden_dim * num_subbands[i],
                    kernel_size=(fc_dim, 1),
                    groups=num_subbands[i]
                    ), # out: (b, k*h, 1, t)
                nn.Conv2d(
                    in_channels=hidden_dim * num_subbands[i],
                    out_channels=self.bands_ranges[i]*2,  # bw * num_subband
                    kernel_size=(1, 1),
                    groups=num_subbands[i]
                    ), # out: (b, ns*2*bw, 1, t)
                Rearrange('b (ns_bw c) 1 t -> b ns_bw t c', c=2) # 덩이가 먼저
                # ns = num_subbands[i]
                # Rearrange('b (bw2 ns) 1 t -> b bw2 ns t'),
                # Rearrange('b (bw c) ns t -> b bw ns t c', c=2)
                )
             for i in range(len(num_subbands))])
        
        # self.mlp_list = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(fc_dim, hidden_dim),
        #         nn.Linear(hidden_dim, bw)
        #     ) for bw in self.width_subbands
        # ])
        
    
    def forward(self, q):
        # q: (b n k t)
        q_bands = [q[:, :, self.num_subbands_cum[i]:self.num_subbands_cum[i+1]] for i in range(len(self.num_subbands))]
        outs = []
        for q_band, ln, mlp in zip(q_bands, self.ln_list, self.mlp_list):
            q_band = rearrange(q_band, 'b n k t -> b k n t') # n: fc_dim
            q_band = ln(q_band)
            m_band = mlp(q_band)  # (b f t c)
            outs.append(m_band)
        mask = torch.cat(outs, dim=-3) # (b, f, t, c)
        
        return mask
            
            
            
if __name__=='__main__':
    sec = 4
    sr = 44100
    nfft=2048
    hop=512
    lr=None
    bs=None
    
    bsrnn = BSRNN(sr, nfft, hop, lr, bs)
    wav = torch.randn(4, sr*sec)
    
    out_wav, out_spec = bsrnn(wav)
    print('out wav:', out_wav.shape)
    print('out spec:', out_spec.shape)
    