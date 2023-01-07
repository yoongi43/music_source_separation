import torch
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from music_source_separation.pl_codes.dataset import MusdbDataModule



from utils import dotdict
import argparse
import yaml
import os

""" 
WTD:
1. Channels?
2. demucs.wav.gev_musdb_wav_dataset() --> train set mixture? automix.py??
    --> Dataset 따로 짜기 ??
    - openunmix
    - limitaug/limitaug.py musdb_train_dataset (changbin)
    - https://github.com/jhtonyKoo/music_mixing_style_transfer/blob/main/mixing_style_transfer/mixing_manipulator/common_audioeffects.py (koo)
    - https://github.com/jhtonyKoo/music_mixing_style_transfer/blob/main/mixing_style_transfer/data_loader/data_loader.py#L115 (koo)
3. Test process
    - Test dataset
    - Metrics(uSDR, cSDR)
    - logging spectrogram
    - lr scheduler
    - valid loop
4. Checkpoint 저장?
5. Wandb?
"""

DEMUCS_CONFIG_PATH = './demucs/conf/config.yaml'

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data-dir', type=str, default='/data1/yoongi/musdb')
    """Model Configs"""
    parser.add_argument('--model', type=str, default='bsrnn', choices=['bsrnn'])
    parser.add_argument('--target-stem', type=str, default='vocals', choices=['drums', 'bass', 'other', 'vocals'])
    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--hop-length', type=int, default=512)
    
    parser.add_argument('--bands', type=int, nargs='+', default=[1000, 4000, 8000, 16000, 20000])
    parser.add_argument('--num-subbands', type=int, nargs='+', default=[10, 12, 8, 8, 2, 1])
    parser.add_argument('--num-band-seq-module', type=int, default=12) # 12
    
    """Training Configs"""
    # parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=100)
    # parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--ckpt-dir', type=str, default='ckpt/')
    
    """Dataset"""
    parser.add_argument('--data-dir', type=str, default='/data1/yoongi/musdb/')
    
    args=parser.parse_args()
    return args


def main(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(list(map(str, args.gpus)))
    
    with open(DEMUCS_CONFIG_PATH) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        
    if args.model=='bsrnn':
        from music_source_separation.bsrnn.model_pl import BSRNN
        model = BSRNN(target_stem=args.target_stem,
                      sr=configs['dset']['musdb_samplerate'],
                      n_fft=args.n_fft,
                      hop_length=args.hop_length,
                      lr = args.lr,
                      batch_size=args.batch_size,
                      num_band_seq_module=args.num_band_seq_module,
                      bands=args.bands,
                      num_subbands=args.num_subbands)
    else:
        NotImplementedError
        
    datamodule = MusdbDataModule(data_dir=args.data_dir, batch_size=args.batch_size, target=args.target_stem, data_num=2)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    logger = WandbLogger(project='music_source_separation')
    
    
    trainer = Trainer(
        accelerator='gpu', #'auto'
        devices = len(args.gpus),
        amp_backend='native',
        precision=16,
        max_epochs=args.max_epochs,
        gradient_clip_val=5,
        logger=logger,
        # val_check_interval=3,
        check_val_every_n_epoch=3,
        callbacks=[TQDMProgressBar(refresh_rate=1)]
    )
    os.makedirs(args.ckpt_dir, exist_ok=True)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, datamodule=datamodule)
    
    
if __name__=="__main__":
    args = parse_args()
    main(args)
    