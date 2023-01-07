from bsrnn.model import *
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