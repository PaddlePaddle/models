# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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
from utils import melspect
import config as c
import librosa
import numpy as np
import sys
import os
import glob
if __name__ == '__main__':
    wavs = glob.glob(c.esc50_audio+'/*.wav')
    print('{} wavs listed'.format(len(wavs)))
    dst_folder = c.esc50_mel
    os.makedirs(dst_folder,exist_ok=True)
    for i,f in enumerate(wavs):
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
        np.save(os.path.join(dst_folder,f.split('/')[-1][:-4]),x)
