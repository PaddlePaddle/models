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

__all__ = [
    'set_backend',
    'get_backends',
    'resample',
    'to_mono',
    'depth_convert',
    'normalize',
    'save_wav',
    'load',
]
import os
import warnings
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import resampy
import soundfile as sf
from numpy import ndarray as array
from scipy.io import wavfile

from ..utils import ParameterError
from ._ffmpeg import DecodingError, FFmpegAudioFile

NORMALMIZE_TYPES = ['linear', 'gaussian']
MERGE_TYPES = ['ch0', 'ch1', 'random', 'average']
RESAMPLE_MODES = ['kaiser_best', 'kaiser_fast']
SUPPORT_BACKENDS = ['ffmpeg', 'soundfile']

EPS = 1e-8

BACK_END = None


def set_backend(backend: Union[str, None] = 'ffmpeg'):
    """Set audio decoding backend.
    Parameters:
        backend(str|None): The name of the backend to use. If None, paddleaudio will
            choose the optimal backend automatically.

    Notes:
        Use get_backends() to get available backends.

    """
    global BACK_END
    if backend and backend not in SUPPORT_BACKENDS:
        raise ParameterError(f'Unsupported backend {backend} ,' +
                             f'supported backends are {SUPPORT_BACKENDS}')
    BACK_END = backend


def get_backends():
    return SUPPORT_BACKENDS


def _safe_cast(y: array, dtype: Union[type, str]) -> array:
    """Data type casting in a safe way, i.e., prevent overflow or underflow.
    Notes:
        This function is used internally.
    """
    return np.clip(y, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)


def _ffmpeg_load(file: os.PathLike,
                 offset: Optional[float] = None,
                 duration: Optional[int] = None) -> Tuple[array, int]:
    """Load audio file using audioread ffmpeg backend.
    Notes:
        This function is for internal use only.
    """
    with FFmpegAudioFile(file) as f:
        sr = f.samplerate
        buffer = b''
        for d in f.read_data():
            buffer += d
    wav = np.frombuffer(buffer, dtype='int16')
    if f.channels != 1:
        wav = wav.reshape((
            -1,
            f.channels,
        )).transpose(1, 0)
    if offset:
        wav = wav[int(offset * sr):]
    if duration is not None:
        frame_duration = int(duration * sr)
        wav = wav[:frame_duration]

    return wav, sr


def _sound_file_load(file: os.PathLike,
                     offset: Optional[float] = None,
                     dtype: str = 'int16',
                     duration: Optional[int] = None) -> Tuple[array, int]:
    """Load audio using soundfile library.
    This function loads audio file using libsndfile.

    Reference:
        http://www.mega-nerd.com/libsndfile/#Features
    Notes:
        This function is for internal use only.
    """
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


def _sox_file_load():
    """Load audio using sox library.
    This function loads audio file using sox.

    Reference:
        http://sox.sourceforge.net/
    Notes:
        This function is for internal use only.
    """
    raise NotImplementedError()


def depth_convert(y: array, dtype: Union[type, str]) -> array:
    """Convert audio array to target dtype safely.
    The function converts audio waveform to a target dtype, with addition steps of
    preventing overflow/underflow and preserving audio range.

    Parameters:
        y(array): the input audio array of shape [n,], [1,n] or [2,n].
        dtype(str|type): the target dtype. The following dtypes are supported:
            'int16', 'int8', 'float32' and 'float64'.
    """

    SUPPORT_DTYPE = ['int16', 'int8', 'float32', 'float64']
    if y.dtype not in SUPPORT_DTYPE:
        raise ParameterError(
            f'Unsupported audio dtype, ' +
            f'y.dtype is {y.dtype}, supported dtypes are {SUPPORT_DTYPE}')

    if dtype not in SUPPORT_DTYPE:
        raise ParameterError(
            f'Unsupported audio dtype, ' +
            f'target dtype  is {dtype}, supported dtypes are {SUPPORT_DTYPE}')

    if dtype == y.dtype:
        return y

    if dtype == 'float64' and y.dtype == 'float32':
        return _safe_cast(y, dtype)
    if dtype == 'float32' and y.dtype == 'float64':
        return _safe_cast(y, dtype)

    if dtype == 'int16' or dtype == 'int8':
        if y.dtype in ['float64', 'float32']:
            factor = np.iinfo(dtype).max
            y = np.clip(y * factor,
                        np.iinfo(dtype).min,
                        np.iinfo(dtype).max).astype(dtype)
            y = y.astype(dtype)
        else:
            if dtype == 'int16' and y.dtype == 'int8':
                factor = np.iinfo('int16').max / np.iinfo('int8').max - EPS
                y = y.astype('float32') * factor
                y = y.astype('int16')

            else:  # dtype == 'int8' and y.dtype=='int16':
                y = y.astype('int32') * np.iinfo('int8').max / \
                    np.iinfo('int16').max
                y = y.astype('int8')

    if dtype in ['float32', 'float64']:
        org_dtype = y.dtype
        y = y.astype(dtype) / np.iinfo(org_dtype).max
    return y


def resample(y: array,
             src_sr: int,
             target_sr: int,
             mode: str = 'kaiser_fast') -> array:
    """Apply audio resampling to the input audio array.

     Notes:
        1. This function uses resampy.resample to do the resampling.
        2. The default mode is kaiser_fast.  For better audio quality,
            use mode = 'kaiser_best'
     """
    if mode == 'kaiser_best':
        warnings.warn(
            f'Using resampy in kaiser_best to {src_sr}=>{target_sr}.' +
            f'This function is pretty slow, ' +
            f'we recommend the mode kaiser_fast in large scale audio training')

    if not isinstance(y, np.ndarray):
        raise TypeError(
            f'Only support numpy array, but received y in {type(y)}')

    if mode not in RESAMPLE_MODES:
        raise ParameterError(f'resample mode must in {RESAMPLE_MODES}')

    return resampy.resample(y, src_sr, target_sr, filter=mode)


def to_mono(y: array, merge_type: str = 'ch0') -> array:
    """Convert stereo audio to mono audio.
    Parameters:
        y(array): the input audio array of shape [2,n], where n is the number of audio samples.
        merge_type(str): the type of algorithm for mergin. Supported types are
            "average": the audio samples from both channels are averaged.
            "ch0": all audio samples from channel 0 are taken as output.
            "ch1: all audio samples from channel 1 are taken as output.
            "random": all audio samples from channel 0 or 1 are taken as output.
        The default value is "average".
    Returns:
        The mono (single-channel) audio.
    Notes:
        This function will keep the audio dtype and will automatically handle the averaging precision
        for int16 or int8 dtype.
    """
    if merge_type not in MERGE_TYPES:
        raise ParameterError(
            f'Unsupported merge type {merge_type}, available types are {MERGE_TYPES}'
        )
    if y.ndim > 2:
        raise ParameterError(
            f'Unsupported audio array,  y.ndim > 2, the shape is {y.shape}')
    if y.ndim == 1:  # nothing to merge
        return y

    if merge_type == 'ch0':
        return y[0]
    if merge_type == 'ch1':
        return y[1]
    if merge_type == 'random':
        return y[np.random.randint(0, 2)]

    # need to do averaging according to dtype

    if y.dtype == 'float32':
        y_out = y.mean(0)
    elif y.dtype == 'int16':
        y_out = y.mean(0)
        y_out = np.clip(y_out,
                        np.iinfo(y.dtype).min,
                        np.iinfo(y.dtype).max).astype(y.dtype)
    elif y.dtype == 'int8':
        y_out = y.mean(0)
        y_out = np.clip(y_out,
                        np.iinfo(y.dtype).min,
                        np.iinfo(y.dtype).max).astype(y.dtype)
    else:
        raise ParameterError(f'Unsupported dtype: {y.dtype}')
    return y_out


def normalize(y: array,
              norm_type: str = 'linear',
              mul_factor: float = 1.0) -> array:
    """Normalize the input audio.

     Parameters:
        norm_type(str): normalization algorithm. Supported types are
            'linear': the audio is normalized linearly such that np.max(np.abs(y))==mul_factor
            'gaussian': the audio is normalized such that np.mean(y)==0 and np.std(y)==mul_factor
            The default value is 'linear'.
        norm_mul_factor(float): additional multiplication factor after normalization.
            The default value is 1.0.
    Notes:
        The audio will be converted to float32, unless its dtype is originly float64.
    """
    if y.dtype not in ['float32', 'float64']:
        y = y.astype('float32')

    if norm_type == 'linear':
        amax = np.max(np.abs(y))
        factor = 1.0 / (amax + EPS)
        y = y * factor * mul_factor
    elif norm_type == 'gaussian':
        amean = np.mean(y)
        astd = np.std(y)
        astd = max(astd, EPS)
        y = mul_factor * (y - amean) / astd
    else:
        raise NotImplementedError(f'norm_type should be in {NORMALMIZE_TYPES}')

    return y


def save_wav(y: array, sr: int, file: os.PathLike) -> None:
    """Save audio file to disk.
    This function saves audio to disk using scipy.io.wavfile, with additional step
    to convert input waveform to int16 unless it already is int16.

    Parameters:
        y(array): the audio data.
        sr(int|None): the sample rate of the audio data. If sr does not match the actual audio data,
        the resulting file will encounter play-back problems.
    Notes:
        The function only supports raw wav format.
    """
    if y.ndim == 2 and y.shape[0] > y.shape[1]:
        warnings.warn(
            f'The audio array tried to saved has {y.shape[0]} channels ' +
            f'and the wave length is {y.shape[1]}. It\'s that what you mean?' +
            f'If not, try to tranpose the array before saving.')
    if not file.endswith('.wav'):
        raise ParameterError(
            f'only .wav file supported, but dst file name is: {file}')

    if sr <= 0:
        raise ParameterError(
            f'Sample rate should be larger than 0, recieved sr = {sr}')

    if y.dtype not in ['int16', 'int8']:
        warnings.warn(
            f'input data type is {y.dtype}, will convert data to int16 format before saving'
        )
        y_out = depth_convert(y, 'int16')
    else:
        y_out = y

    wavfile.write(file, sr, y_out.T)


def load(
        file: os.PathLike,
        sr: Optional[int] = None,
        mono: bool = True,
        merge_type: str = 'average',  # ch0,ch1,random,average
        normal: bool = True,
        norm_type: str = 'linear',
        norm_mul_factor: float = 1.0,
        offset: float = 0.0,
        duration: Optional[int] = None,
        dtype: str = 'float32',
        resample_mode: str = 'kaiser_fast') -> Tuple[array, int]:
    """Load audio file from disk.
    This function loads audio from disk using using automatically chosen backend.
    Parameters:
        file(os.PathLike): the path of the file. URLs are not supported.
        sr(int|None): the target sample rate after loaded. If None, the original (native)
            sample rate is deduced from the file itself and no resampling is performed.
            If the native sample rate is different from specified target sample rate, resamping
            is performed according to resample_mode parameter.
            The default value is None.
        mono(bool): whether to convert audio to mono using algorithm specified in merge_type parameter
            if it is originally steore. See to_mono() for more details.
            The default value is True.
        merge_type(str): the merging algorithm. See to_mono() for more details.
            The default value is 'ch0'.
        normal(bool): whether to normalize the audio waveform. If True, the audio will be normalized using algorithm
            specified in norm_type. See normalize() for more details.
            The default value is True.
        norm_mul_factor(float): additional multiplication factor for normalization. See normalize() for more details.
            The default value is 1.0.
        norm_type(str): normalization algorithm. Supported types are 'linear' and 'gaussian'. See normalize() for
            more details. The default value is 'linear'.
        offset(float): the time (in seconds) for offseting the audio after loaded, e.g., set offset=1.0 to load all data
            after 1.0 second. If the audio duration is less than offset, empty array is returned.
            The default value is 0.
        duration(float): the audio length measured in seconds after it is loaded. If None, or the actual audio duration is
            less than specified duration, the actual audio array is returned without padding.
            The default value is None.
        dtype(str): the target dtype of the return audio array. The dynamic range of audio samples will be
            adjusted according to dtype.
        resample_mode(str): the algorithm used in resampling. See resample() for more details.

    Raises:
        FileNotFoundError, if audio file is not found
        DecodingError, if audio file is not supported

    """
    if BACK_END == 'ffmpeg':
        y, r = _ffmpeg_load(file, offset=offset, duration=duration)
    elif BACK_END == 'soundfile':
        y, r = _sound_file_load(file,
                                offset=offset,
                                dtype=dtype,
                                duration=duration)
    else:
        try:
            y, r = _sound_file_load(file,
                                    offset=offset,
                                    dtype=dtype,
                                    duration=duration)
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Trying to load a file that doesnot exist {file}')
        except:
            try:
                y, r = _ffmpeg_load(file, offset=offset, duration=duration)
            except DecodingError:
                raise DecodingError(f'Failed to load and decode file {file}')

    if not ((y.ndim == 1 and len(y) > 0) or (y.ndim == 2 and len(y[0]) > 0)):
        return np.array([], dtype=dtype)  # return empty audio

    if mono:
        y = to_mono(y, merge_type)

    if sr is not None and sr != r:
        y = resample(y, r, sr, mode=resample_mode)
        r = sr

    if normal:
        y = normalize(y, norm_type, norm_mul_factor)
    elif dtype in ['int8', 'int16']:
        # still need to do normalization, before depth convertion
        y = normalize(y, 'linear', 1.0)

    y = depth_convert(y, dtype)
    return y, r
