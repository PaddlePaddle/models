import argparse
import glob
import os

import h5py
import numpy as np
import tqdm

import paddleaudio as pa

#from pylab import *
parser = argparse.ArgumentParser(description='wave2mel')
parser.add_argument('--wav_file', type=str, required=False, default='')
parser.add_argument('--wav_list', type=str, required=False, default='')
parser.add_argument('--wav_h5_file', type=str, required=False, default='')
parser.add_argument('--wav_h5_list', type=str, required=False, default='')
parser.add_argument('--output_folder', type=str, required=False, default='./')
parser.add_argument('--output_h5', type=bool, required=False, default=True)
parser.add_argument('--sample_rate', type=int, required=False, default=32000)
parser.add_argument('--window_size', type=int, required=False, default=1024)
parser.add_argument('--mel_bins', type=int, required=False, default=128)
parser.add_argument('--hop_length', type=int, required=False, default=640)  #20ms
parser.add_argument('--fmin', type=int, required=False, default=50)  #25ms
parser.add_argument('--fmax', type=int, required=False, default=16000)  #25ms
args = parser.parse_args()
#args.wav_h5_file = '/ssd2/laiyongquan/audioset/h5/audioset_unblance_group28.h5'

assert not (args.wav_h5_file == '' and args.wav_h5_list == ''\
and args.wav_list == '' and args.wav_file == ''), 'one of wav_file,wav_list,\
wav_h5_file,wav_h5_list needs to specify'

if args.wav_h5_file != '':
    h5_files = [args.wav_h5_file]
if args.wav_h5_list != '':
    h5_files = open(args.wav_h5_list).read().split('\n')
    h5_files = [h for h in h5_files if len(h.strip()) != 0]

dst_folder = args.output_folder
print(f'{len(h5_files)} h5 files listed')
for f in h5_files:
    print(f'processing {f}')
    dst_file = os.path.join(dst_folder, f.split('/')[-1])
    print(f'target file {dst_file}')
    assert not os.path.exists(dst_file), f'target file {dst_file} existed'
    src_h5 = h5py.File(f)
    dst_h5 = h5py.File(dst_file, "w")
    for key in tqdm.tqdm(src_h5.keys()):
        s = src_h5[key][:]
        s = pa.depth_convert(s, 'float32')
        # s = pa.resample(s,32000,args.sample_rate)
        x = pa.features.mel_spect(s,
                                  sample_rate=args.sample_rate,
                                  window_size=args.window_size,
                                  hop_length=args.hop_length,
                                  mel_bins=args.mel_bins,
                                  fmin=args.fmin,
                                  fmax=args.fmax,
                                  window='hann',
                                  center=True,
                                  pad_mode='reflect',
                                  ref=1.0,
                                  amin=1e-10,
                                  top_db=None)
        #         figure(figsize=(8,8))
        #         imshow(x)
        #         show()
        #         print(x.shape)

        dst_h5.create_dataset(key, data=x)
    src_h5.close()
    dst_h5.close()
