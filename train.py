from tqdm import tqdm
from utils import augment_modules, adjust_seq_len, apply_model_overlap_add, cal_metrics, new_sdr
from utils import get_spectrogram, rms_normalize_torch, db2linear, linear2db
# from metrics import museval_sdr, cmgan_snr

import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange
import wandb
import numpy as np
import random
import os ; opj=os.path.join
from glob import glob
from natsort import natsorted
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import pyloudnorm as pyln

class Solver:
    def __init__(self,
                 args,
                 configs_demucs,
                 model,
                 device,
                 optimizer,
                 train_loader,
                 valid_loader=None,
                 test_loader=None,
                 scheduler=None,
                 augmentation=True
                 ):
        stem_list = ['mix', 'drums', 'bass', 'other', 'vocals']
        stem_dict = {stem:i for i,stem in enumerate(stem_list)}
        self.target_stem_idx = stem_dict[args.target_stem]        
        
        self.args = args
        self.cfg_dmcs = configs_demucs
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device=device
        if self.device==torch.device('cuda:0'):
            self.model = nn.DataParallel(self.model)
            
        if augmentation:
            self.augments = augment_modules(args=configs_demucs).to(device)
        else:
            self.augments = nn.Identity().to(device)
        self.model.to(device)
        # self.loss_fn = nn.L1Loss(reduction='sum')
        self.loss_fn = nn.L1Loss(reduction='mean')
        self.meter = pyln.Meter(args.sr)
        
    def train(self):
        start_epoch = self.args.start_epoch+1 if self.args.resume else 0
        for epoch in range(start_epoch, self.args.max_epochs):
            self.train_epoch(epoch=epoch)
            if epoch % self.args.valid_per == 0:
                if self.valid_loader is not None:
                    self.valid_epoch(epoch=epoch)
            
            if epoch % self.args.ckpt_per == 0:
                try:
                    state_dict = self.model.module.state_dict()
                except AttributeError:
                    state_dict = self.model.state_dict()
                ckpt = dict(net = state_dict, opt=self.optimizer.state_dict())
                torch.save(ckpt, opj(self.args.save_dir, 'ckpt', str(epoch).zfill(4)+'.pt'))
                ckpt_list = natsorted(glob(opj(self.args.save_dir, 'ckpt', '*')))
                if len(ckpt_list) > 4:
                    os.remove(ckpt_list[0])
                

    def train_epoch(self, epoch):
        self.model.train()
        scaler = torch.cuda.amp.GradScaler()
        metrics = dict(csdr=[], usdr=[])
        for idx, batch in enumerate(tqdm(self.train_loader, desc=f'Train Epoch: {epoch}')): 
            self.optimizer.zero_grad()   
            
            sources = self.augments(batch.to(self.device))
            
            mix_wav = sources.sum(dim=1)  # (b, src, c t) -> (b c t)
            target_wav = sources[:, self.target_stem_idx-1]  # no 'mix' index
            mix_wav = adjust_seq_len(mix_wav, hop_length=self.args.hop_length)
            target_wav = adjust_seq_len(target_wav, hop_length=self.args.hop_length)
            
            """NORMALIZE"""
            
            ### RMS normalize
            # mix_wav, gain = rms_normalize_torch(mix_wav, ref_dBFS=-10, return_gain=True)
            # target_wav = target_wav * gain
            ### LUFS normalize
            # target_lufs = -14.
            # loudness = self.meter.integrated_loudness(mix_wav)
            # gain = target_lufs - loudness
            # target_wav = target_wav * db2linear(gain)
            # mix_wav = mix_wav * db2linear(gain)
            #### mix_wav = pyln.normalize.loudness(mix_wav, loudness, target_lufs)
            """Normalize"""
            
            target_wav_ = rearrange(target_wav, 'b c t -> (b c) t')
            target_spec = \
                torch.stft(target_wav_, n_fft=self.args.n_fft, hop_length=self.args.hop_length, window=torch.hann_window(self.args.n_fft).to(self.device), return_complex=False)
            target_spec = rearrange(target_spec, '(b c) f t ri -> b c f t ri', b=mix_wav.size(0))
            
            with torch.cuda.amp.autocast():
                est_spec, est_wav, est_mask = self.model(mix_wav) # est wav: (b c t), est_spec: (b c f t ri)
                
                loss_spec = self.loss_fn(est_spec, target_spec)
                loss_time = self.loss_fn(est_wav, target_wav)
                # loss_nsdr = - new_sdr(references=target_wav, estimates=est_wav).mean()
                loss_total = self.args.lambda_spec*loss_spec + (1-self.args.lambda_spec)*loss_time
                # loss_total = self.args.lambda_spec*loss_spec + (1-self.args.lambda_spec)*loss_time + loss_nsdr
                # loss_total = self.args.lambda_spec*loss_spec + (1-self.args.lambda_spec)*loss_time + torch.exp(loss_nsdr)
                
            """ METRICS """ 
            ## 시간상 batch중에서 한개만
            metrics_results = cal_metrics(ref=target_wav[0][None], est=est_wav[0][None], sr=self.args.sr, chunks=1)
            csdr = metrics_results['csdr']
            usdr = metrics_results['usdr']
            # nsdr = metrics_results['nsdr']
            metrics['csdr']+=list(csdr)
            metrics['usdr']+=list(usdr)
            # metrics['nsdr']+=list(nsdr)
            
            lr_cur = self.optimizer.state_dict()['param_groups'][0]['lr']
            
            scaler.scale(loss_total).backward()
            # clipping
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=5.)
            # 
            scaler.step(self.optimizer)
            scaler.update()
                
            """ Logging """
            if not self.args.debug:
                wandb.log({'epoch': epoch,
                           'learning rate': lr_cur,
                            'train loss spec': loss_spec,
                            'train loss time': loss_time,
                            # 'train loss nsdr': loss_nsdr,
                            'train loss total': loss_total,
                            'train metric csdr': np.mean(csdr).item(),
                            'train metric usdr': np.mean(usdr).item()
                            })
            
            if self.args.debug:
                print('loss_spec: ', loss_spec.item(), 'loss_time: ', loss_time.item(), 'loss_total: ', loss_total.item(),
                      'cur_lr: ', lr_cur, 'epoch: ', epoch, 
                      'csdr: ', csdr, 'usdr: ', usdr
                      )
                if idx>2: ## For Wandb debugging, unindent
                    break          
        if self.scheduler is not None:
            self.scheduler.step()
        
        
        if epoch % self.args.wandb_per==0:
            if not self.args.debug:
                """ Audio log"""
                target_wav = rearrange(target_wav, 'b c t -> b t c').detach().cpu().numpy()[0]
                est_wav = rearrange(est_wav, 'b c t -> b t c').detach().cpu().numpy()[0]
                mix_wav = rearrange(mix_wav, 'b c t -> b t c').detach().cpu().numpy()[0]
                est_mask = est_mask.detach().cpu().numpy()[0]
                wandb.log({'Train target wav': wandb.Audio(target_wav, sample_rate=self.args.sr),
                           'Train est wav': wandb.Audio(est_wav, sample_rate=self.args.sr),
                           'Train mix wav': wandb.Audio(mix_wav, sample_rate=self.args.sr)})
                
                """ Spec log """
                fig = get_spectrogram(clean=target_wav, pred=est_wav, mix=mix_wav,
                                    mask=est_mask, sr=self.args.sr, n_fft=self.args.n_fft,
                                    hop_length=self.args.hop_length, batch_idx=None)
                wandb.log({'Train fig': wandb.Image(fig)})
                plt.close(fig)
            
        # if not self.args.debug:
        #     wandb.log({'train metric csdr epoch': np.mean(metrics['csdr']).item(),
        #             'train metric usdr epoch': np.mean(metrics['usdr']).item()})
        
            
            
    def valid_epoch(self, epoch):
        self.model.eval()
        if not self.args.debug:
            if epoch==0:
                return 
        with torch.no_grad():
            metrics=dict(csdr=[], usdr=[])
            random_idx = random.sample(list(range(len(self.valid_loader))), 3)
            for idx, batch in enumerate(tqdm(self.valid_loader, desc=f'Valid Epoch: {epoch}')):
                # if idx not in random_idx:
                #     continue
                mix_wav = batch[:, 0].to(self.device)
                # sources = sources[:, 1:]
                target_wav = batch[:, self.target_stem_idx]
                est_wav = apply_model_overlap_add(model=self.model, mix=mix_wav, sr=self.args.sr,
                                                shifts=1.5, segments=3)
                target_wav = target_wav[...,:est_wav.shape[-1]]  
                """ METRICS """ 
                metrics_results = cal_metrics(ref=target_wav, est=est_wav, sr=self.args.sr, chunks=1)
                csdr = metrics_results['csdr']
                usdr = metrics_results['usdr']
                metrics['csdr']+=list(csdr)
                metrics['usdr']+=list(usdr)
                
                if not self.args.debug:
                    # wandb.log({'valid metric csdr batch': csdr,
                    #         'valid metric usdr batch': usdr})
                    pass
                    
                if self.args.debug:
                    print('### VALID###')
                    print('epoch: ', epoch, 'csdr: ', csdr, 'usdr: ', usdr)
                    if idx>5:
                        break
                    
            if not self.args.debug:
                wandb.log({'valid metric csdr epoch': np.mean(metrics['csdr']).item(),
                        'valid metric usdr epoch': np.mean(metrics['usdr']).item()})
                
            # target_wav = rearrange(target_wav[0], 'c t -> t c').detach().cpu().numpy()
            # est_wav = rearrange(est_wav[0], 'c t -> t c').detach().cpu().numpy()
            # if not self.args.debug:
            #     wandb.log({"Valid Target sample":wandb.Audio(target_wav, sample_rate=self.args.sr)})
            #     wandb.log({"Valid Estimated sample": wandb.Audio(est_wav, sample_rate=self.args.sr)})
                
            
            
            
            
                    
                
                


                
            
    
    # def train_step(self, batch):
    #     pass
    
    # def valid_step(self, batch):
    #     pass
    
