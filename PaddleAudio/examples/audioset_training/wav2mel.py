import argparse
import glob
import os

import h5py
import numpy as np
import paddle
import paddleaudio as pa
import tqdm
from paddleaudio.functional import melspectrogram

parser = argparse.ArgumentParser(description='wave2mel')
parser.add_argument('--wav_file', type=str, required=False, default='')
parser.add_argument('--wav_list', type=str, required=False, default='')
parser.add_argument('--wav_h5_file', type=str, required=False, default='')
parser.add_argument('--wav_h5_list', type=str, required=False, default='')
parser.add_argument('--output_folder', type=str, required=False, default='./')
parser.add_argument('--output_h5', type=bool, required=False, default=True)
parser.add_argument('--dst_h5_file', type=str, required=False, default='')

parser.add_argument('--sample_rate', type=int, required=False, default=32000)
parser.add_argument('--window_size', type=int, required=False, default=1024)
parser.add_argument('--mel_bins', type=int, required=False, default=128)
parser.add_argument('--hop_length', type=int, required=False,
                    default=640)  #20ms
parser.add_argument('--fmin', type=int, required=False, default=50)  #25ms
parser.add_argument('--fmax', type=int, required=False, default=16000)  #25ms
parser.add_argument('--skip_existed', type=int, required=False,
                    default=1)  #25ms

args = parser.parse_args()

assert not (args.wav_h5_file == '' and args.wav_h5_list == ''\
and args.wav_list == '' and args.wav_file == ''), 'one of wav_file,wav_list,\
wav_h5_file,wav_h5_list needs to specify'

h5_files = []
wav_files = []
if args.wav_h5_file != '':
    h5_files = [args.wav_h5_file]
elif args.wav_h5_list != '':
    h5_files = open(args.wav_h5_list).read().split('\n')
    h5_files = [h for h in h5_files if len(h.strip()) != 0]
elif args.wav_list != '':
    wav_files = open(args.wav_list).read().split('\n')
    wav_files = [h for h in wav_files if len(h.strip()) != 0]

elif args.wav_file != '':
    wav_files = [args.wav_file]

dst_folder = args.output_folder

if len(h5_files) > 0:
    print(f'{len(h5_files)} h5 files listed')
    for f in h5_files:
        print(f'processing {f}')
        dst_file = os.path.join(dst_folder, f.split('/')[-1])
        print(f'target file {dst_file}')
        if args.skip_existed != 0 and os.path.exists(dst_file):
            print(f'skipped file {f}')
            continue
        assert not os.path.exists(dst_file), f'target file {dst_file} existed'
        src_h5 = h5py.File(f)
        dst_h5 = h5py.File(dst_file, "w")
        for key in tqdm.tqdm(src_h5.keys()):
            s = src_h5[key][:]
            s = pa.depth_convert(s, 'float32')
            # s = pa.resample(s,32000,args.sample_rate)

            x = melspectrogram(paddle.to_tensor(s),
                               sr=args.sample_rate,
                               win_length=args.window_size,
                               n_fft=args.window_size,
                               hop_length=args.hop_length,
                               n_mels=args.mel_bins,
                               f_min=args.fmin,
                               f_max=args.fmax,
                               window='hann',
                               center=True,
                               pad_mode='reflect',
                               to_db=True,
                               amin=1e-3,
                               top_db=None)

            dst_h5.create_dataset(key, data=x[0].numpy())
        src_h5.close()
        dst_h5.close()

if len(wav_files) > 0:

    assert args.dst_h5_file != '', 'for using wav file or wav list, dst_h5_file must be specified'

    dst_file = args.dst_h5_file
    assert not os.path.exists(dst_file), f'target file {dst_file} existed'
    dst_h5 = h5py.File(dst_file, "w")
    print(f'{len(wav_files)} wav files listed')
    for f in tqdm.tqdm(wav_files):
        s, _ = pa.load(f, sr=args.sample_rate)
        x = melspectrogram(paddle.to_tensor(s),
                           sr=args.sample_rate,
                           win_length=args.window_size,
                           n_fft=args.window_size,
                           hop_length=args.hop_length,
                           n_mels=args.mel_bins,
                           f_min=args.fmin,
                           f_max=args.fmax,
                           window='hann',
                           center=True,
                           pad_mode='reflect',
                           to_db=True,
                           amin=1e-3,
                           top_db=None)
        #         figure(figsize=(8,8))
        #         imshow(x)
        #         show()
        #         print(x.shape)
        key = f.split('/')[-1][:11]
        dst_h5.create_dataset(key, data=x[0].numpy())
    dst_h5.close()
