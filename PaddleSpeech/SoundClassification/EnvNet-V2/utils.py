import time
import random
import math
import numpy as np


# Default data augmentation
def padding(pad):
    """
    Padding function.
    """

    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f


def random_crop(size):
    """
    Randomly cropping function. 
    """

    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start:start + size]

    return f


def normalize(factor):
    """
    Normilzing function.
    """

    def f(sound):
        return sound / factor

    return f


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    """
    Randomly scaling function.
    """

    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))
        return scaled_sound

    return f


def random_gain(db):
    """
    Random gain function.
    """

    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


# For testing phase
def multi_crop(input_length, n_crops):
    """
    A function to crop a audio sample into n_crops segments.
    """

    def f(sound):
        stride = (len(sound) - input_length) // (n_crops - 1)
        sounds = [
            sound[stride * i:stride * i + input_length] for i in range(n_crops)
        ]
        return np.array(sounds)

    return f


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (
        2 * np.log10(12194) + 2 * np.log10(freq_sq) -
        np.log10(freq_sq + 12194**2) - np.log10(freq_sq + 20.6**2) - 0.5 *
        np.log10(freq_sq + 107.7**2) - 0.5 * np.log10(freq_sq + 737.9**2))
    weight = np.maximum(weight, min_db)
    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    """
    Compute audio gain.
    """
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    #MOHAIMEN: no xrange anymore
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i:i + n_fft]**2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i:i + n_fft])
            power_spec = np.abs(spec)**2
            a_weighted_spec = power_spec * np.power(10,
                                                    a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    """
    Mix two sounds for bc learning.
    """
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t**2 + (1 - t)**2))
    return sound


class Timer(object):
    """
    Calculate runing speed and estimated time of arrival(ETA)
    """

    def __init__(self, total_step: int):
        self.total_step = total_step
        self.last_start_step = 0
        self.current_step = 0
        self._is_running = True

    def start(self):
        self.last_time = time.time()
        self.start_time = time.time()

    def stop(self):
        self._is_running = False
        self.end_time = time.time()

    def count(self) -> int:
        if not self.current_step >= self.total_step:
            self.current_step += 1
        return self.current_step

    @property
    def timing(self) -> float:
        run_steps = self.current_step - self.last_start_step
        self.last_start_step = self.current_step
        time_used = time.time() - self.last_time
        self.last_time = time.time()
        return run_steps / time_used

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def eta(self) -> str:
        if not self.is_running:
            return '00:00:00'
        scale = self.total_step / self.current_step
        remaining_time = (time.time() - self.start_time) * scale
        return seconds_to_hms(remaining_time)


def seconds_to_hms(seconds: int) -> str:
    """
    Convert the number of seconds to hh:mm:ss
    """
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = '{:0>2}:{:0>2}:{:0>2}'.format(h, m, s)
    return hms_str


def compute_eval_acc(y_pred, y_target, n_crops=10):
    # Reshape y_pred to shape it like each sample comtains 10 samples.
    y_pred = y_pred.reshape(y_pred.shape[0] // n_crops, n_crops,
                            y_pred.shape[1])

    # Calculate the average of class predictions for 10 crops of a sample
    y_pred = np.mean(y_pred, axis=1)

    # Get the indices that has highest average value for each sample
    y_pred = y_pred.argmax(axis=1)

    # Doing the samething for y_target
    y_target = (y_target.reshape(y_target.shape[0] // n_crops, n_crops,
                                 y_target.shape[1])).mean(axis=1).argmax(axis=1)

    accuracy = (y_pred == y_target).mean()
    loss = np.mean((y_target - y_pred)**2)
    return accuracy, loss
