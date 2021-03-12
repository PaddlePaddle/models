import paddle
import numpy as np
import config as c
import librosa
from utils import melspect
from ipdb import set_trace
from pylab import *

paddle.seed(100)
np.random.seed(100)

from paddle.io import Dataset, DataLoader, IterableDataset


def get_labels(file_list):
    labels =  [f.split('/')[-1].split('-')[-1].split('.')[0] for f in file_list]
    return [int(l) for l in labels]


def spect_augment(spect,max_time_mask = 3,
        max_freq_mask = 3,
        max_time_mask_width = 30,
        max_freq_mask_width = 20):
    nt,nf= spect.shape
    assert(nf==64)
    num_time_mask = int(paddle.randint(0,high=max_time_mask))
    num_freq_mask = int(paddle.randint(0,high=max_freq_mask))

    time_mask_width = int(paddle.randint(0,high=max_time_mask_width))
    freq_mask_width = int(paddle.randint(0,high=max_freq_mask_width))
    #print(num_time_mask)
    #print(num_freq_mask)


    for i in range(num_time_mask):
        start = int(paddle.randint(0,high=nt-time_mask_width))
        spect[start:start+time_mask_width,:]=0
    for i in range(num_freq_mask):
        start = int(paddle.randint(0,high=nf-freq_mask_width))
        spect[:,start:start+freq_mask_width]=0
        
  #  figure
   # imshow(spect)
    #show()
    return spect

def random_crop(s):
    n = len(s)
    idx = int(paddle.randint(0,high=n-sample_len))
    #idx = np.random.randint(0,high=n-sample_len)
    return s[idx:idx+sample_len]
sample_len = c.mel_sample_len

def random_crop2d(s):
    n = len(s)
   # idx = int(paddle.randint(0,high=n-sample_len))
    idx = np.random.randint(0,high=n-sample_len)
    return s[idx:idx+sample_len]    
    
def random_split(esc_file_list,test_fold):
    
   # test_fold = np.random.randint(5)
    val_fold = np.random.randint(5)+1
    while val_fold == test_fold:
        val_fold = np.random.randint(5)+1
    
    print('test fold:{},val fold:{}'.format(test_fold,val_fold))
    assert(len(esc_file_list)==2000)
    
    test_files = []
    train_files = []
    val_files = []
    
    for file in esc_file_list:
        if '/{}-'.format(test_fold) in file:
            test_files += [file]
        elif '/{}-'.format(val_fold) in file:
            val_files += [file]
        else:
            train_files += [file]
    
    return train_files,val_files,test_files



class Esc50(Dataset):
    def __init__(self, files,labels,crop=True):
        super(Esc50, self).__init__()
        self.files = files
        self.labels = labels
        self.crop = crop
        
    def __getitem__(self, idx):

        x = np.load(self.files[idx])
        if self.crop:
            x = random_crop2d(x)
            #print(x.shape)
            x = spect_augment(x)
#         figure
        return x, self.labels[idx]
    def __len__(self):
        return len(self.files)
    
def get_loaders(test_fold,seed=100):
    np.random.seed(seed) 
    file_manifest = '/ssd3/public/datasets/ESC50/esc50/wav16_mel.list'
    wavs = open(file_manifest).read().split('\n')[:-1]
    train_files,val_files,test_files = random_split(wavs,test_fold)
    train_files += val_files
    train_labels = get_labels(train_files)
    val_labels = get_labels(val_files)
    test_labels = get_labels(test_files)
    train_dataset = Esc50(train_files,train_labels,True)
    val_dataset = Esc50(val_files,val_labels,False)
    test_dataset = Esc50(test_files,test_labels,False)

    num_class = c.num_class
    batch_size = c.batch_size
    train_loader = DataLoader(train_dataset,  shuffle=True, #places=PLACE,
                              batch_size=batch_size, drop_last=False,
                              num_workers=0, use_shared_memory=False)

    val_loader = DataLoader(val_dataset,  shuffle=True, 
                            batch_size=batch_size, drop_last=False,
                            num_workers=0, use_shared_memory=False)

    test_loader = DataLoader(test_dataset,  shuffle=True, 
                            batch_size=batch_size, drop_last=False,
                            num_workers=0, use_shared_memory=False)

    return train_loader,val_loader,test_loader
