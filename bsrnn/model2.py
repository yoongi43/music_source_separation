from bsrnn.model import *
from bsrnn.model import _quantize_hz2bins
from bsrnn.conformer import ConformerBlock

""" 변형된 모델들 """
class BSConformer(nn.Module):
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
            *[BandSeqConformerModule(channels=channels, fc_dim=fc_dim, group=group, num_subbands=num_subbands) for _ in range(num_band_seq_module)]
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
        return est_spec, est_wav
    

class BSRNN_Skip(nn.Module):
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
        assert num_band_seq_module > 1
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
        self.bandseq_module = nn.ModuleList(
            [BandSeqModelingModule(channels=channels, fc_dim=fc_dim, group=group, num_subbands=num_subbands) for _ in range(num_band_seq_module)]
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
        for idx, mod in enumerate(self.bandseq_module):
            if idx==0:
                q = mod(z) + z
            else:
                q = mod(q) + q
        mask = self.maskest_module(q+z)  # 실질적인 model output.
        
        cspec = torch.view_as_complex(spec)
        cmask = torch.view_as_complex(mask)
        est_cspec = cmask * cspec
        est_wav = torch.istft(est_cspec, n_fft=self.n_fft, hop_length=self.hop_length, window=torch.hann_window(self.n_fft).to(wav.device)) #((b c) t)
        est_spec = torch.view_as_real(est_cspec)  # ((b c) f t ri)
        est_wav = rearrange(est_wav, '(b c) t -> b c t', c=self.channels)
        est_spec = rearrange(est_spec, '(b c) f t ri -> b c f t ri', c=self.channels)
        return est_spec, est_wav
    
    
class BSRNN_Overlap(nn.Module):
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
        self.bandsplit_module = BandSplitModule_Overlap(sr=sr, n_fft=n_fft, channels=channels, fc_dim=fc_dim, bands=bands, num_subbands=num_subbands)
        self.bandseq_module = nn.Sequential(
            *[BandSeqModelingModule_Overlap(channels=channels, fc_dim=fc_dim, group=group, num_subbands=num_subbands) for _ in range(num_band_seq_module)]
        )
        self.maskest_module = MaskEstimationModule_Overlap(sr=sr, n_fft=n_fft, channels=channels, fc_dim=fc_dim, bands=bands, num_subbands=num_subbands)
        
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
        return est_spec, est_wav

    
    
""" Modules"""
class BandSeqConformerModule(nn.Module):
    def __init__(self,
                 fc_dim,
                 channels=2,
                 group=16,
                 num_subbands=[10, 12, 8, 8, 2, 1]):
        super().__init__()
        fc_dim = channels * fc_dim
        group = channels * group
        hidden_dim = fc_dim  # BLSTM -> hiddendim = fc_dim 에서 2곱해짐
        num_total_subbands = sum(num_subbands)
        self.blstm_seq = nn.Sequential(  # rnn across T
            Rearrange('b n k t -> (b k) n t'),
            nn.GroupNorm(num_groups=group, num_channels=fc_dim), # n/group  # out:((b, k), n, t) 
            Rearrange('(b k) n t -> (b k) t n', k=num_total_subbands), # (batch, seq, hidden)
            ConformerBlock(dim=hidden_dim, dim_head=hidden_dim//4, heads=4, attn_dropout=0.2, ff_dropout=0.2),
            nn.Linear(hidden_dim, fc_dim),
            Rearrange('(b k) t n -> b n k t', k=num_total_subbands)
        )
        self.blstm_band = nn.Sequential(  # rnn across K
            Rearrange('b n k t -> (b t) n k'),
            nn.GroupNorm(num_groups=group, num_channels=fc_dim),
            Rearrange('b_t n k -> b_t k n'),
            ConformerBlock(dim=hidden_dim, dim_head=hidden_dim//4, heads=4, attn_dropout=0.2, ff_dropout=0.2), # out: (b, k, hidden)
            nn.Linear(hidden_dim, fc_dim)
        )
        
    def forward(self, z):
        # z: (b n k t)
        z_hat = self.blstm_seq(z) + z
        q = self.blstm_band(z_hat)
        q = rearrange(q, '(b t) k n -> b n k t', b=z_hat.size(0))
        q = q + z_hat
        return q
    

class BandSplitModule_Overlap(nn.Module):
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
        self.band_intervals = self.bands[2:]-self.bands[:-2]

        # self.ln_list = nn.ModuleList([nn.GroupNorm(1, band_interval) for band_interval in self.band_intervals])
        # self.fc_list = nn.ModuleList([nn.Linear(band_interval, fc_dim) for band_interval in self.band_intervals])
        self.layer_list = nn.ModuleList([
            nn.Sequential(
                # nn.GroupNorm(1, band_interval),  # LayerNorm
                ### Layernorm in F. not T.
                Rearrange('b f t c -> b t (f c)', c=2*channels),
                nn.LayerNorm(band_interval*channels*2), # with Rearrange('b f t c -> b t (f c)')
                # Rearrange('b f t c -> b t (f c)', c=2*channels),
                nn.Linear(2*channels*band_interval, channels*fc_dim),
                Rearrange('b t n -> b n 1 t')
                )
            for band_interval in self.band_intervals])
        
    def forward(self, cspec):
        # cspec: (b, f, t, (channel, ri))  # mono or stereo
        cspec_bands = [cspec[:, self.bands[i]:self.bands[i+2]] for i in range(len(self.bands)-2)]
        outs = []
        for cspec_band, layer in zip(cspec_bands, self.layer_list):
            out = layer(cspec_band)
            outs.append(out) # out:(b, n, 1, t)
        outs = torch.cat(outs, dim=-2) # outs: (b, n, k, t)
        return outs
    
    
class BandSeqModelingModule_Overlap(nn.Module):
    def __init__(self,
                 fc_dim,
                 channels=2,
                 group=16,
                 num_subbands=[10, 12, 8, 8, 2, 1]):
        super().__init__()
        fc_dim = channels * fc_dim
        group = channels * group
        hidden_dim = fc_dim  # BLSTM -> hiddendim = fc_dim 에서 2곱해짐
        num_total_subbands = sum(num_subbands)-1  # overlapped
        self.blstm_seq = nn.Sequential(  # rnn across T
            Rearrange('b n k t -> (b k) n t'),
            nn.GroupNorm(num_groups=group, num_channels=fc_dim), # n/group
            Rearrange('(b k) n t -> (b k) t n', k=num_total_subbands), # (batch, seq, hidden)
            nn.LSTM(input_size=fc_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True), # out:(b, t, hidden_dim*2(bi))
            ExtractOutput(),
            nn.Linear(2*hidden_dim, fc_dim),
            Rearrange('(b k) t n -> b n k t', k=num_total_subbands)
        )
        self.blstm_band = nn.Sequential(  # rnn across K
            Rearrange('b n k t -> (b t) n k'),
            nn.GroupNorm(num_groups=group, num_channels=fc_dim),
            Rearrange('b_t n k -> b_t k n'),
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
    
    
class MaskEstimationModule_Overlap(nn.Module):
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
        self.band_intervals = self.bands[2:]-self.bands[:-2]
        num_total_subbands = len(self.bands)-2
        
        self.channels = channels
        fc_dim=fc_dim*channels
        hidden_dim = 4*fc_dim # *4
        self.layer_list = nn.ModuleList([
            nn.Sequential(
                Rearrange('b n t -> b t n'),
                nn.LayerNorm(fc_dim),
                nn.Linear(fc_dim, hidden_dim),
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
            
        # Overlap / Add for frequency bins.
        # Band intervals 개수만큼의 overlap.
        # mask_real = torch.cat(outs, dim=-3)
        bc, f_, t, ri = out.shape
        mask_real = torch.zeros((bc, self.bands[-1], t, ri)).to(out.device)
        for i in range(len(outs)-2):
            f_start = self.bands[i]
            f_middle = self.bands[i+1]
            f_end = self.bands[i+2]
            # print('i', i)
            # print('outshape', out[i].shape)
            # print('freqs', f_start, f_middle, f_end)
            if i==0:
                mask_real[:, f_start:f_end] += outs[i]
            else:
                mask_real[:, f_start:f_middle] = (mask_real[:, f_start:f_middle] + outs[i][:, :f_middle-f_start])/2
                mask_real[:, f_middle:f_end] = mask_real[:, f_middle:f_end] + outs[i][:, f_middle-f_start:f_end-f_start]
        print('mask real size:', mask_real.size())
        return mask_real