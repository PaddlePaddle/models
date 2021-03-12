
from utils import melspect
import config as c
import librosa
import numpy as np
import sys

if __name__ == '__main__':
   # dst_folder = '/ssd3/public/datasets/ESC50/esc50/mel/'
    #file_manifest = '/ssd3/public/datasets/ESC50/esc50/wav16.list'
    for i,f in enumerate(wavs):

        if len(sys.argv)!=3:
            print('usage: python preprocess_esc50.py file_manifest dst_folder')
            return
        file_manifest = sys.argv[1]
        dst_folder = sys.argv[2]
        
        wavs = open(file_manifest).read().split('\n')[:-1]
        print('{} wavs listed'.format(len(wavs)))
        s,_ = librosa.load(f,sr=c.sample_rate)
        x = melspect(s,
         sample_rate=c.sample_rate,
         window_size = c.window_size,
         hop_size=c.hop_size,
         mel_bins=c.mel_bins,
        fmin=c.fmin,
         fmax=c.fmax,
         window='hann', 
         center=True, 
         pad_mode='reflect',
         ref=1.0,
         amin=1e-10, 
         top_db=None
        )
        if i %100 == 0:
            print('{}/{}'.format(i,len(wavs)))
        np.save(dst_folder+f.split('/')[-1][:-4]+'.mel',x)
    os.system('cp config.py {}'.format(dst_folder))


        
        
