# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import h5py
import json
import numpy as np
import os
import paddle
from paddle.io import Dataset, DataLoader, IterableDataset
from paddleaudio import augmentation 
import subprocess
import warnings
import yaml
from utils import get_ytid_clsidx_mapping,get_logger,get_labels527

with open('./config.yaml') as F:
    c = yaml.load(F,Loader=yaml.FullLoader)
logger = get_logger(__name__,os.path.join(c['log_path'],'log.txt'))

def spect_permute(spect,tempo_axis,nblocks):
    """spectrogram  permutaion"""
    assert spect.ndim == 2., 'only supports 2d tensor or numpy array'
    if tempo_axis == 0:
        nt,nf= spect.shape
    else:
        nf,nt= spect.shape
    if nblocks <= 1:
        return spect
    
    block_width = nt // nblocks + 1
    if tempo_axis == 1:
        blocks = [spect[:,block_width*i:(i+1)*block_width] for i in range(nblocks)]
        np.random.shuffle(blocks)
        new_spect = np.concatenate(blocks,1)
    else:
        blocks = [spect[block_width*i:(i+1)*block_width,:] for i in range(nblocks)]
        np.random.shuffle(blocks)
        new_spect = np.concatenate(blocks,0)
    return new_spect
        
    

#precompute the mapping   
logger.info('precomputing mapping for ytid and clsidx ')
ytid2clsidx,clsidx2ytid = get_ytid_clsidx_mapping()
logger.info('done')
    
def random_choice(a):
    i = paddle.randint(0,high=len(a),shape=(1,))
    return a[int(i)]

class H5AudioSetSingle(Dataset):
    """
    dataset class for wraping a single h5 file. 
    This class is used by H5Audioset
    """
    def __init__(self,h5_file,balance_sampling=True,padding=False):
        super(H5AudioSetSingle, self).__init__()
        
        self.h5_file = h5_file
        self.balance_sampling =balance_sampling
        self.h5_data = h5py.File(h5_file)
        in_h5 = dict([(k[:11],True) for k in self.h5_data.keys()])
        filter_keys = lambda keys: [k for k in keys if in_h5.get(k,False)]
        self.local_keys = [k[:11] for k in list(self.h5_data.keys())] # size of ytid is 11
        self.clsidx2keys_local = {}
        n_samples = []
        no_sample_class = []

        for k in range(c['num_classes']):
            keys = filter_keys(clsidx2ytid[k])
            self.clsidx2keys_local.update({k:keys})
            if len(keys)==0:
                no_sample_class += [k]
            n_samples += [len(keys)]
            
        
       

        logger.info('cls {} has no audio samples'.format(no_sample_class)) 
        self.n_average = int(np.mean(n_samples))
        self.n_min = np.min(n_samples)
        self.padding = padding
        
            
        
    def __getitem__(self,idx):
        if self.balance_sampling:
            keys = []
            cls_id = idx%c['num_classes']
            keys = self.clsidx2keys_local[int(cls_id)]
            while len(keys)==0:
                cls_id = int(paddle.randint(0,527,(1,)))
                keys = self.clsidx2keys_local[int(cls_id)]
            k = random_choice(keys)
            cls_ids = ytid2clsidx[k]
        else:
            k = self.local_keys[idx]
            cls_ids = ytid2clsidx[k]
        
        y = np.zeros((c['num_classes'],),'float32')
        for l in cls_ids:
            y[l] = 1.0
        x = self.h5_data[k][:,:]    
        if self.padding:
            if x.shape[1] <= c['max_mel_len']:
                pad_width = c['max_mel_len'] - x.shape[1]+1
                x = np.pad(x,((0,0),(pad_width//2,pad_width//2+1)))
            x = x[:,:c['max_mel_len']]
            
        return x.T,y
        
        
    def __len__(self,):
        if self.balance_sampling:
            return self.n_average*527
        else:
            return len(self.local_keys)
    def __repr__(self,):
        return self.h5_file
def get_keys(h5_files):
    keys = []
    n_per_h5 = []
    for f in h5_files:
        with h5py.File(f) as F:
            n_per_h5 += [len(F.keys())]
            keys += list(F.keys())
    return keys,n_per_h5
                
                
class H5AudioSet(Dataset):
    """
    Dataset class for Audioset, with mel features stored in multiple hdf5 files.
    The h5 files store mel-spectrogram features pre-extracted from wav files. 
    Use wav2mel.py to do feature extraction.
   
    """
    def __init__(self,
                 h5_files,
                 augment=True,
                 training=True,
                 balance_sampling=True):
        super(H5AudioSet, self).__init__()
        self.h5_files = h5_files 
        self.all_keys,self.n_per_h5 = get_keys(h5_files)
        self.n_per_h5_cumsum = np.cumsum([0]+self.n_per_h5)
        self.current_h5_idx = -1
        self.augment = augment
        self.training = training
        self.balance_sampling = balance_sampling
        print(f'{len(self.h5_files)} h5 files, totally {len(self.all_keys)} audio files listed')
      
    def shuffle(self,):
        np.random.shuffle(self.h5_files)
    def _process(self,x):
        if self.training:
            if x.shape[1] <= c['mel_crop_len']:
                pad_width = (c['mel_crop_len'] - x.shape[1])//2+1
                x = np.pad(x,((0,0),(pad_width,pad_width)))
        else:
            if x.shape[1] <= c['max_mel_len']: #
                pad_width = c['max_mel_len'] - x.shape[1]+1
                x = np.pad(x,((0,0),(0,pad_width)))
            x = x[:,:501]

        if self.augment:
            x = augmentation.random_crop2d(x,c['mel_crop_len'],tempo_axis=1)
            
            x = spect_permute(x,tempo_axis = 1,nblocks=random_choice([0,2,3]))
            
            aug_level = random_choice([0.2,0.1,0])
            x = augmentation.adaptive_spect_augment(x,tempo_axis=1,level=aug_level)
        return x.T

    def __getitem__(self, idx):
        
        h5_idx = np.argwhere(self.n_per_h5_cumsum <= idx)[-1][0]
        h5_idx = h5_idx % len(self.h5_files)
        if h5_idx != self.current_h5_idx:
            self.current_h5_idx = h5_idx
            logger.info('loading h5 file '+self.h5_files[h5_idx])
            self.h5_dataset = H5AudioSetSingle(self.h5_files[h5_idx],
                                               balance_sampling=self.balance_sampling)
        s,labels = self.h5_dataset[idx] # any number is ok
        x = self._process(s.T)
       
       
        return x, labels
    def __len__(self):
    
        return len(self.all_keys)
    
    
def get_ytid2labels(segment_csv):
    """
    compute the mapping (dict object) from youtube id to audioset labels. 
    """
    with open(segment_csv) as F:
        lines = F.read().split('\n')
    
    lines = [l for l in lines if len(l)>0 and l[0]!='#'  ]
    ytid2labels = {l.split(',')[0]:l.split('"')[-2] for l in lines}
    return ytid2labels

class AudioSet(Dataset):
    """
    Regular Dataset class for Audioset.
    It supports loading wav files or mel feature fiels stored in a given folder
    
    """
    def __init__(self, 
                 segment_csv,
                 data_folder,
                 data_type = 'mel', # mel or wav
                 augment=False,
                 training=True):
        super().__init__()
        
       
        labels527 = get_labels527()
        ytid2labels = get_ytid2labels(segment_csv)
        label2clsidx = {l:i for i,l in enumerate(labels527)}
        if data_type == 'mel':
            train_files = glob.glob(data_folder+'/*.npy')
        elif data_type == 'wav':
            train_files = glob.glob(data_folder+'/*.wav')
        else:
            assert data_type in ['mel','wav'], 'data_type must be mel or wav'
            
        assert len(train_files)!=0, f'{data_folder} seems to be empty (no *.npy or *.wav files found)'
       
        train_files_ytid = [f.split('/')[-1][:11] for f in train_files] # size of youtube id is 11
        train_labels = [ytid2labels[ytid] for ytid in train_files_ytid]
        train_clsidx = [[label2clsidx[l] for l in label.split(',')] for label in train_labels]

        # save params
        self.data_type = data_type
        self.train_files = train_files
        self.train_clsidx = train_clsidx
        self.augment = augment
        self.training = training
    def _pad(self,x,pad_len):
        if x.shape[1] <= pad_len:
            w = (pad_len - x.shape[1])//2+1
            x = np.pad(x,((0,0),(w,w)))

        
    def _load(self,file):
       
        if self.data_type == 'wav':
            s,_ = paddleaudio.load(file,sr=c['sample_rate'])
            s = np.pad(s, ((0,1),(0,0)), 'constant', constant_values=(0,))
            power = (np.exp(s)-1)**2
            power = np.abs(s)**2
            melW = librosa.filters.mel(sr=c['sample_rate'],
                                       n_fft=c['window_size'],
                                       n_mels=c['mel_bins'],
                                       fmin=c['fmin'], 
                                       fmax=c['fmax'])
            mel = np.matmul(melW,power)
            x = librosa.power_to_db(mel,ref=1.0,amin=1e-10,top_db=None)
        else:
            s = np.load(file)
            print(s.shape)
            x = s

        if self.training:
            x = self._pad(x,c['mel_crop_len'])
            if self.augment:
                x = augmentation.random_crop2d(x,c['mel_crop_len'],tempo_axis=1)
                x = augmentation.spect_augment(x,tempo_axis=1)
        else: #use all data for evaluation
            x = self._pad(x,c['max_mel_len'])
            x = x[:,:c['max_mel_len']]

        return x.T

    def __getitem__(self, idx):
        file = self.train_files[idx]
        x = self._load(file)
        labels = self.train_clsidx[idx]
        y = np.zeros((c['num_classes'],),'float32')
        for l in labels:
            y[l] = 1.0
        return x, y
    def __len__(self):
        return len(self.train_files)

def get_loader():
    
    train_h5_files = glob.glob(c['unbalance_train_h5'])
    train_h5_files += [c['balance_train_h5']]
    
    train_dataset =  H5AudioSet(train_h5_files,
                                balance_sampling=True,
                                augment=True,
                                training=True)
    
    val_dataset = H5AudioSetSingle(c['balance_eval_h5'],
                        balance_sampling=False,padding=True)
    
    train_loader = DataLoader(train_dataset, 
                              shuffle=False, # must be false
                              batch_size=c['batch_size'], 
                              drop_last=True,
                              num_workers=c['num_workers'],
                              use_buffer_reader=True,
                              use_shared_memory=True)

    val_loader = DataLoader(val_dataset,  
                            shuffle=False, 
                            batch_size=c['val_batch_size'], 
                            drop_last=False,
                            num_workers=0)
    
    return train_loader, val_loader
    
if __name__ == '__main__':
    train_h5_files = glob.glob('./audioset/mel-128/*.h5')
    dataset =  H5AudioSet(train_h5_files,balance_sampling=True,augment=True,training=True)
    x,y = dataset[0]
    print(x.shape,y.shape)
    dataset = H5Dataset(c['balance_eval_h5'],
                        balance_sampling=False,padding=True)
    x,y = dataset[0]
    print(x.shape,y.shape)
    
    
