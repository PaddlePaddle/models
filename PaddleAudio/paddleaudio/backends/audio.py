# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import warnings

import numpy as np
from scipy.io import wavfile

try:
    import librosa
    has_librosa = True
except:
    has_librosa = False

try:
    import soundfile as sf
    has_snf = True
except:
    has_snf = False
try:
    import resampy
    has_resampy = True
except:
    has_resampy = False

__norm_types__ = ['linear', 'gaussian']
__mono_types__ = ['ch0', 'ch1', 'random', 'average']

__all__ = ['resample', 'to_mono', 'depth_convert', 'normalize', 'save', 'load']


def resample(y, src_sr, target_sr):

    warnings.warn(
        f'Using resampy to {src_sr}=>{target_sr}. This function is pretty slow, we recommend to process audio using ffmpeg'
    )
    assert type(y) == np.ndarray, 'currently only numpy data are supported'
    assert type(
        src_sr) == int and src_sr > 0 and src_sr <= 48000, 'make sure type(sr) == int and sr > 0 and sr <= 48000,'
    assert type(
        target_sr
    ) == int and target_sr > 0 and target_sr <= 48000, 'make sure type(sr) == int and sr > 0 and sr <= 48000,'

    if has_resampy:
        return resampy.resample(y, src_sr, target_sr)

    if has_librosa:
        return librosa.resample(y, src_sr, target_sr)

    assert False, 'requires librosa or resampy to do resampling, pip install resampy'


def to_mono(y, mono_type='average'):

    assert type(y) == np.ndarray, 'currently only numpy data are supported'

    if mono_type not in __mono_types__:
        assert False, 'Unsupported mono_type {}, available types are {}'.format(mono_type, __mono_types__)

    if y.ndim == 1:
        return y
    if y.ndim > 2:
        assert False, 'Unsupported audio array,  y.ndim > 2, the shape is {}'.format(y.shape)
    if mono_type == 'ch0':
        return y[0]
    if mono_type == 'ch1':
        return y[1]
    if mono_type == 'random':
        return y[np.random.randint(0, 2)]

    if y.dtype == 'float32':
        return (y[0] + y[1]) * 0.5
    if y.dtype == 'int16':
        y1 = y.astype('int32')
        y1 = (y1[0] + y1[1]) // 2
        y1 = np.clip(y1, np.iinfo(y.dtype).min, np.iinfo(y.dtype).max).astype(y.dtype)
        return y1
    if y.dtype == 'int8':
        y1 = y.astype('int16')
        y1 = (y1[0] + y1[1]) // 2
        y1 = np.clip(y1, np.iinfo(y.dtype).min, np.iinfo(y.dtype).max).astype(y.dtype)
        return y1

    assert False, 'Unsupported audio array type,  y.dtype={}'.format(y.dtype)


def __safe_cast__(y, dtype):
    return np.clip(y, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)


def depth_convert(y, dtype):  # convert audio array to target dtype

    assert type(y) == np.ndarray, 'currently only numpy data are supported'

    __eps__ = 1e-5
    __supported_dtype__ = ['int16', 'int8', 'float32', 'float64']
    if y.dtype not in __supported_dtype__:
        assert False, 'Unsupported audio dtype,  y.dtype is {}, supported dtypes are {}'.format(
            y.dtype, __supported_dtype__)
    if dtype not in __supported_dtype__:
        assert False, 'Unsupported dtype,  target dtype is {}, supported dtypes are {}'.format(
            dtype, __supported_dtype__)

    if dtype == y.dtype:
        return y

    if dtype == 'float64' and y.dtype == 'float32':
        return __safe_cast__(y, dtype)
    if dtype == 'float32' and y.dtype == 'float64':
        return __safe_cast__(y, dtype)

    if dtype == 'int16' or dtype == 'int8':
        if y.dtype in ['float64', 'float32']:
            factor = np.iinfo(dtype).max
            y = np.clip(y * factor, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
            y = y.astype(dtype)
        # figure
        # plot(y)
        # show()
        else:
            if dtype == 'int16' and y.dtype == 'int8':
                factor = np.iinfo('int16').max / np.iinfo('int8').max - __eps__
                y = y.astype('float32') * factor
                y = y.astype('int16')

            else:  #dtype == 'int8' and y.dtype=='int16':
                y = y.astype('int32') * np.iinfo('int8').max / np.iinfo('int16').max
                y = y.astype('int8')

    if dtype in ['float32', 'float64']:
        org_dtype = y.dtype
        y = y.astype(dtype) / np.iinfo(org_dtype).max
    return y


def sound_file_load(file, offset=None, dtype='int16', duration=None):
    with sf.SoundFile(file) as sf_desc:
        sr_native = sf_desc.samplerate
        if offset:
            sf_desc.seek(int(offset * sr_native))
        if duration is not None:
            frame_duration = int(duration * sr_native)
        else:
            frame_duration = -1
        y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T

    return y, sf_desc.samplerate


def normalize(y, norm_type='linear', mul_factor=1.0):

    assert type(y) == np.ndarray, 'currently only numpy data are supported'

    __eps__ = 1e-8
    #set_trace()
    if norm_type == 'linear':
        # amin = np.min(y)
        amax = np.max(np.abs(y))
        factor = 1.0 / (amax + __eps__)
        y = y * factor * mul_factor
    elif norm_type == 'gaussian':
        amean = np.mean(y)
        mul_factor = max(0.01, min(mul_factor, 0.2))
        astd = np.std(y)
        y = mul_factor * (y - amean) / (astd + __eps__)
    else:
        assert False, 'not implemented error, norm_type should be in {}'.format(__norm_types__)

    return y


def save(y, sr, file):
    assert type(y) == np.ndarray, 'currently only numpy data are supported'
    assert type(sr) == int and sr > 0 and sr <= 48000, 'make sure type(sr) == int and sr > 0 and sr <= 48000,'

    if y.dtype not in ['int16', 'int8']:
        warnings.warn('input data type is {}, saving data to int16 format'.format(y.dtype))
        yout = depth_convert(y, 'int16')
    else:
        yout = y

    wavfile.write(file, sr, y)


def load(
        file,
        sr=None,
        mono=True,
        mono_type='average',  # ch0,ch1,random,average
        normal=True,
        norm_type='linear',
        norm_mul_factor=1.0,
        offset=0.0,
        duration=None,
        dtype='float32'):

    if has_librosa:
        y, r = librosa.load(file, sr=sr, mono=False, offset=offset, duration=duration,
                            dtype='float32')  #alwasy load in float32, then convert to target dtype

    elif has_snf:
        y, r = sound_file_load(file, offset=offset, dypte=dtype, duration=duration)

    else:
        assert False, 'not implemented error'

    ##
    assert (y.ndim == 1 and len(y) > 0) or (y.ndim == 2 and len(y[0]) > 0), 'audio file {} looks empty'.format(file)

    if mono:
        y = to_mono(y, mono_type)

    if sr is not None and sr != r:
        y = resample(y, r, sr)
        r = sr

    if normal:
        #    print('before nom',np.max(y))
        y = normalize(y, norm_type, norm_mul_factor)
    # print('after norm',np.max(y))
    #plot(y)
    #show()
    if dtype in ['int8', 'int16'] and (normalize == False or normalize == True and norm_type == 'guassian'):
        y = normalize(y, 'linear', 1.0)  # do normalization before converting to target dtype

    y = depth_convert(y, dtype)
    #figure
    #plot(y)
    #show()
    return y, r
