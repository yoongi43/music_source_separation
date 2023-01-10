import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
import os
import argparse
import yaml
from utils import DotConfig
from tqdm import tqdm

# from demucs.demucs.train import get_datasets2
from demucs.demucs.wav import get_musdb_wav_datasets
from utils import augment_modules
from train import Solver
import wandb
from datetime import datetime
import os ; opj=os.path.join
from glob import glob

DEMUCS_CONFIG_PATH = './demucs/conf/config.yaml'

"""
WTD:
1. DDP
2. Spectrogram?
3. save .wav (eavluate code)
4. uSDR(SNR)???
"""


def parse_args():
    parser = argparse.ArgumentParser()
    """Model Configs (BSRNN) """
    parser.add_argument('--model', type=str, choices=['bsrnn', 'bsconformer', 'bsrnn_skip', 'bsrnn_overlap'])
    parser.add_argument('--target-stem', type=str, default='vocals', choices=['drums', 'bass', 'other', 'vocals'])
    # parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--hop-length', type=int, default=512)
    parser.add_argument('--channels', type=int, default=1)  # mono:1, stereo:2
    
    parser.add_argument('--bands', type=int, nargs='+', default=[1000, 4000, 8000, 16000, 20000])
    parser.add_argument('--num-subbands', type=int, nargs='+', default=[10, 12, 8, 8, 2, 1])
    parser.add_argument('--num-band-seq-module', type=int, default=12) # 12
    
    """Training setup"""
    parser.add_argument('--batch-size', type=int, default=8)  # original: gpu당 2개
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay', type=float, default=0.98)
    parser.add_argument('--lambda-spec', type=float, default=0.5)
    parser.add_argument('--valid-per', type=int, default=5)
    parser.add_argument('--ckpt-per', type=int, default=5)
    parser.add_argument('--max-epochs', type=int, default=100)
    # parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])  # original : 8개
    # parser.add_argument('--ckpt-dir', type=str, default='ckpt/')
    parser.add_argument('--num-workers', type=int, default=16)
    
    """ Dataset """
    parser.add_argument('--data-dir', type=str, default='/data1/yoongi/musdb/')

    """ Resume """
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--nowstr', type=str, default=None)
    parser.add_argument('--start-epoch', type=int, default=5)
    
    
    """ETC"""
    parser.add_argument('--base-dir', type=str, default='./result')
    
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--debug-wandb', action='store_true', default=False)
    
    
    """WANDB"""
    parser.add_argument('--project', type=str, default='music_source_separation')
    parser.add_argument('--wandb-id',type=str, default=None)
    parser.add_argument('--memo', type=str, default='')

    args = parser.parse_args()
    return args

def main(args):
    """ SETUPS"""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(list(map(str, args.gpus)))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    with open(DEMUCS_CONFIG_PATH) as f:
        cfg_dmcs = yaml.load(f, Loader=yaml.FullLoader)
    # args_dset = dotdict(args_dmcs['dset'])
    cfg_dmcs['dset']['musdb'] = args.data_dir
    cfg_dmcs['dset']['channels'] = args.channels
    cfg_dmcs = DotConfig(cfg_dmcs)
    args.sr = cfg_dmcs.dset.musdb_samplerate
    
    
    
    """ Runs / Directories """
    args.run_name = args.model+'-'+args.target_stem
    if args.nowstr is None:
        """New run"""
        if args.debug:
            save_dir = 'debug'
        else:
            args.nowstr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            save_dir = '_'.join([args.nowstr, args.run_name]) if args.run_name is not None else nowstr
        args.save_dir = opj(args.base_dir, save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        for d in ['train', 'valid', 'test', 'ckpt', 'wandb']:
            os.makedirs(opj(args.save_dir, d), exist_ok=True)
    else:
        """ RESUME"""
        args.resume = True
        ckpt_dirs = sorted(glob(opj(args.base_dir, args.nowstr + '*')))
        assert len(ckpt_dirs)== 1
        args.ckpt_dir = ckpt_dirs[0]
        args.save_dir = args.ckpt_dir
        save_dir = '_'.join([args.nowstr, args.run_name]) if args.run_name is not None else args.nowstr
        # args.save_dir = opj(args.base_dir, save_dir)
        # os.makedirs(args.save_dir, exist_ok=True)
        # for d in ['train', 'valid', 'test', 'ckpt', 'wandb']:
        #     os.makedirs(opj(arg.save_dir, d), exist_ok=True)
        ckpt = torch.load(opj(args.ckpt_dir, 'ckpt', str(args.start_epoch).zfill(4) + '.pt'))

    """ WANDB SETUP """
    if not args.debug:
        # args.group = 
        os.environ['WADNB_START_METHOD'] = 'thread'
        wandb_data = wandb.init(
            project=args.project,
            id = save_dir + str(args.memo) if args.wandb_id is None else args.wandb_id,
            dir=args.save_dir,
            resume=False if args.nowstr is None else True,
            config=args
        )
    
    """ Load Model"""
    model = load_model(args)
        
    if args.resume:
        model.load_state_dict(ckpt['net'])
        print('Pretrained model loaded in epoch: ', args.start_epoch)
        
    """Dataset"""
    train_dataset, valid_dataset = get_musdb_wav_datasets(cfg_dmcs.dset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                              num_workers=args.num_workers)
    print('Train set length: ', len(train_dataset))
    print('Valid set length: ', len(valid_dataset))
    
    """Optimizer, Scheduler"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, 
                                                  lr_lambda=lambda epoch: args.lr_decay**(epoch//2))
    # scheduler = None

    solver = Solver(
        args=args,
        configs_demucs=cfg_dmcs,
        model=model,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=None,
        scheduler=scheduler,
        augmentation=True
    )
    
    solver.train()
    
    
def load_model(args):
    """Load Model"""
    if args.model=='bsrnn':
        from bsrnn.model import BSRNN
        model = BSRNN(target_stem=args.target_stem,
                      sr=args.sr,
                      n_fft=args.n_fft,
                      hop_length=args.hop_length,
                      channels=args.channels,
                      num_band_seq_module=args.num_band_seq_module,
                      bands=args.bands,
                      num_subbands=args.num_subbands)
    elif args.model=='bsconformer':
        from bsrnn.model2 import BSConformer
        model = BSConformer(target_stem=args.target_stem,
                      sr=args.sr,
                      n_fft=args.n_fft,
                      hop_length=args.hop_length,
                      channels=args.channels,
                      num_band_seq_module=args.num_band_seq_module,
                      bands=args.bands,
                      num_subbands=args.num_subbands)
    
    elif args.model=='bsrnn_skip':
        from bsrnn.model2 import BSRNN_Skip
        model = BSRNN_Skip(target_stem=args.target_stem,
                            sr=args.sr,
                            n_fft=args.n_fft,
                            hop_length=args.hop_length,
                            channels=args.channels,
                            num_band_seq_module=args.num_band_seq_module,
                            bands=args.bands,
                            num_subbands=args.num_subbands)
    elif args.model=='bsrnn_overlap':
        from bsrnn.model2 import BSRNN_Overlap
        model = BSRNN_Overlap(target_stem=args.target_stem,
                            sr=args.sr,
                            n_fft=args.n_fft,
                            hop_length=args.hop_length,
                            channels=args.channels,
                            num_band_seq_module=args.num_band_seq_module,
                            bands=args.bands,
                            num_subbands=args.num_subbands)
    
    else:
        NotImplementedError
        
    return model

if __name__=='__main__':
    args = parse_args()
    main(args)