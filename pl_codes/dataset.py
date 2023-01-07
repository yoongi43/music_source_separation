from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import os
import yaml
from demucs.demucs.wav import get_musdb_wav_datasets
from utils import dotdict
from tqdm import tqdm
from music_source_separation.temp.cb_limitaug import musdb_train_Dataset
import soundfile as sf

NUM_WORKERS=int(os.cpu_count() / 2)
DEMUCS_ARGS_PATH = './demucs/conf/config.yaml'

class MusdbDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        target='vocals', 
        num_workers=NUM_WORKERS,
        demucs_args_path=DEMUCS_ARGS_PATH,
        data_num=None
    ):
        super().__init__()
        self.save_hyperparameters()
        with open(demucs_args_path) as f:
            args_demucs = yaml.load(f, Loader=yaml.FullLoader)
        args_dset = dotdict(args_demucs['dset'])
        args_dset.musdb = data_dir
        self.args_dset = args_dset
        
        self.setup()
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = musdb_train_Dataset(target=self.hparams.target, root=self.args_dset.musdb, data_num=self.hparams.data_num)
            # self.valid_set = musdb_valid_dataset(target=self.hparams.target, args=self.args_dset)
            self.valid_set = musdb_train_Dataset(target=self.hparams.target, root=self.args_dset.musdb, split='valid', data_num=self.hparams.data_num)
            
            # self.train_set, self.valid_set = get_musdb_wav_datasets(self.args_dset)
            # mixture(valid) + ['drums', 'bass', 'other', 'vocals']
        if stage == 'test' or stage is None:
            self.test_set = None
            # raise NotImplementedError
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=1, num_workers=self.hparams.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        assert False
        return DataLoader()
    
            

class musdb_valid_dataset(Dataset):
    def __init__(self, args, target='vocals'):
        _, valid_set = get_musdb_wav_datasets(args)
        dur = args.segment
        shift = args.shift
        sr = args.musdb_samplerate
        stem_list = ['mix', 'drums', 'bass', 'other', 'vocals']
        stem_dict = {stem:i for i,stem in enumerate(stem_list)}
        target_stem_idx = stem_dict[target]

        self.mix_list = []
        self.target_list = []
        for data in tqdm(valid_set):
            # data:(5, 2, len)
            audio_len = data.shape[-1]
            print('audio len: ', audio_len/44100/60)
            if data is None:
                continue
            for i in range(5):
                print(data[i].view(-1, 2).shape)
                mono = data[i]
                sf.write(f'./temp/{i}.wav', data[i].view(-1, 2), samplerate=44100)
            assert False
            audio_len = data.shape[-1]
            i = 0
            while True:
                start = i*shift*sr
                end = (i*shift + dur) * sr
                if end >= audio_len:
                    break
                audio = data[..., start:end]
                self.mix_list.append(audio[0])
                self.target_list.append(audio[target_stem_idx])
                i += 1
        
    def __getitem__(self, idx):
        return self.mix_list[idx], self.target_list[idx]
    
    def __len__(self):
        return len(self.mix_list)
        
        
if __name__=="__main__":
    dm = MusdbDataModule(data_dir='/data1/yoongi/musdb',
                         batch_size=2,
                         target='vocals',
                         data_num=2
                         )
    ts = dm.train_set
    vs = dm.valid_set
    td = dm.train_dataloader()
    vd = dm.val_dataloader()
    
    a, b = ts[0]
    print(len(ts))
    print(len(vs))
    print(len(td))
    print(len(vd))
    print(a.shape, b.shape)
    a, b = vs[0]
    print(a.shape, b.shape)
