"""
Collection of classes to help with finding and adding noise to audio
"""
from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()

from io import BytesIO
from scipy import signal
from scipy.signal.signaltools import lfilter
import base64
import json
import linecache
import logging
import numpy as np
import os
import random
import re
import scikits.samplerate
import soundfile
import time
import warnings
import net

from error import SampleRateError, AudioLengthError, AudioFormatError
from error import ClippingWarning, NormalizationWarning

MSEC = 1000
FULL_SCALE = 1.0  # Full-scale reference range for float32 samples.
DEFAULT_NORMALIZATION = -20.0  # default RMS audio normalization in dB.

logger = logging.getLogger(__name__)


class AudioSegment(object):
    """
    Monaural audio segment abstraction.

    Attributes (private):
        _samples (1darray.float32): audio samples
        _sample_rate (scalar): audio sample rate
        _rms_db (float): root mean square of samples in decibels

    Note:
        Instances of this class are designed to be immutable, so any
        method that would transform any of the attributes returns
        a new instance.  Immutability isn't strictly enforced, so try
        not to do anything evil.
    """

    def __init__(self, samples, sample_rate):
        """
        Create audio segment from samples.

        Args:
            :param samples: audio samples [num_samples x num_channels]
            :type samples:  ndarray
            :param sample_rate: audio sample rate
            :type sample_rate: scalar

        Returns:
            AudioSegment

        Raises:
            TypeError: if an array of samples with more than one dimension
                is provided.

        Note:
            This converts samples to float32 internally,
            and creates a new internal copy of the input samples.
            The audio is stored internally with full scale range
            [-FULL_SCALE, +FULL_SCALE]
            Multi-channel audio is not supported.
        """
        self._sample_rate = sample_rate
        self._samples = convert_samples_to_float32(samples)

        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

        # self._rms_db is computed on demand (when self.rms_db is accessed)
        self._rms_db = None

    def __eq__(self, other):
        """
        Return whether two AudioSegments are equal.

        Returns:
            bool
        """
        # Wrong type
        if type(other) is not type(self):
            return False

        # Wrong sample rate
        if self._sample_rate != other._sample_rate:
            return False

        if self._samples.shape != other._samples.shape:
            return False

        # Samples not equal
        if np.any(self._samples != other._samples):
            return False

        return True

    def __ne__(self, other):
        """
        Return whether two AudioSegments are unequal.

        Returns:
            bool
        """
        return not self.__eq__(other)

    def __add__(self, other):
        """
        Add samples from another segment to those of this segment and return
        a new segment (sample-wise addition, not segment concatenation).

        Args:
            :param other: segment containing samples to be added in.
            :type other: AudioSegment

        Returns:
            new_segment (AudioSegment): new segment containing resulting
                samples.

        Raises:
            SampleRateError: If sample rates of segments don't match.
            AudioLengthError: If length of segments don't match.
        """
        if type(self) != type(other):
            raise TypeError("Cannot add segment of different type: {}"
                            .format(type(other)))

        if self._sample_rate != other._sample_rate:
            raise SampleRateError("Sample rates must match to add segments.")
        if len(self._samples) != len(other._samples):
            raise AudioLengthError("Segment lengths must match to add "
                                   "segments.")

        samples = self.samples + other.samples
        return type(self)(samples, sample_rate=self._sample_rate)

    def __len__(self):
        """
        Returns length of segment in samples.
        """
        return self.num_samples

    def __getitem__(self, pos):
        """
        Return a subsegment or sample value.

        Args:
            :param pos:
                boundaries in seconds provided by the slice object.
                - If negative values are provided, they are interpreted
                  as number of seconds from the end of the segment
                  (as negative indices are in Python/numpy).
                - If an ending boundary value is past the end of the segment,
                  the effective ending boundary is snapped back to the actual
                  end of the segment (as in Python/numpy).
            :type pos: slice

        Returns:
            AudioSegment: Returns a new AudioSegment containing subsegment.

        Examples:
            >>> segment = AudioSegment(np.ones(100), sample_rate=8000)
            >>> # subsegment between 1.5--3.0 seconds.
            >>> subsegment = segment[1.5:3.0]
            >>> # subsegment up to 1.0 seconds before end of segment
            >>> subsegment = segment[:-1.0]
        """
        if pos.step is not None:
            raise IndexError("Variable step is not supported.")
        return self.subsegment(pos.start, pos.stop)

    def __repr__(self):
        """
        Return human-readable representation of segment.
        """
        return ("{}: num_samples={}, sample_rate={}, duration={}sec, "
                "rms={}dB".format(
                    type(self), self.num_samples, self.sample_rate,
                    self.length_in_sec, self.rms_db))

    @classmethod
    def make_silence(cls, duration, sample_rate):
        """
        Creates a silent audio segment of the given duration and
        sample rate.

        Args:
            :param duration: length of silence in seconds
            :type duration: scalar
            :param sample_rate: sample rate
            :type  sample_rate: scalar

        Returns:
            silence of the given duration
        """
        samples = np.zeros(int(float(duration) * sample_rate))
        return cls(samples, sample_rate)

    @classmethod
    def concatenate(cls, *segments):
        """
        Concatenate an arbitrary number of audio segments together.

        Args:
            :param *segments: input audio segments
            :type *segments: [AudioSegment]
        """
        # Perform basic sanity-checks.
        N = len(segments)
        if N == 0:
            raise ValueError("No audio segments are given to concatenate.")
        sample_rate = segments[0]._sample_rate
        for segment in segments:
            if sample_rate != segment._sample_rate:
                raise SampleRateError("Can't concatenate segments with "
                                      "different sample rates")
            if type(segment) is not cls:
                raise TypeError("Only audio segments of the same type "
                                "instance can be concatenated.")

        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate)

    @classmethod
    def from_wav_file(cls, wav_file):
        """
        Create AudioSegment from wav file.

        Args:
            :param wav_file: path to wav file
            :type wav_file: basestring

        Returns:
            AudioSegment
        """
        # This scales the resulting float32 samples to [-1,+1] full-scale
        try:
            sndfile = soundfile.SoundFile(wav_file)
        except RuntimeError as exc:
            raise AudioFormatError('{}'.format(exc))

        if sndfile.format != 'WAV':
            raise AudioFormatError(
                'File {} is not a wav file.'.format(wav_file))

        sample_rate = sndfile.samplerate
        samples = sndfile.read(dtype='float32')

        return cls(samples, sample_rate)

    @classmethod
    def slice_from_wav_file(cls, fname, start=None, end=None):
        """ 
        Loads a small section of an audio without having to load
        the entire file into the memory which can be incredibly wasteful.

        Args:
            :param fname: input audio file name
            :type fname: bsaestring
            :param start: start time in seconds (supported granularity is ms)
                If start is negative, it wraps around from the end. If not
                provided, this function reads from the very beginning.
            :type start: float
            :param end: start time in seconds (supported granularity is ms)
                If end is negative, it wraps around from the end. If not
                provided, the default behvaior is to read to the end of the
                file.
            :type end: float

        Returns:
            the specified slice of input audio in the audio.AudioSegment
            format.
        """
        try:
            sndfile = soundfile.SoundFile(fname)
        except RuntimeError as exc:
            raise AudioFormatError('{}'.format(exc))

        if sndfile.format != 'WAV':
            raise AudioFormatError('File {} is not a wav file.'.format(fname))

        sample_rate = sndfile.samplerate
        if sndfile.channels != 1:
            raise TypeError("{} has more than 1 channel.".format(fname))

        duration = float(len(sndfile)) / sample_rate

        if start is None:
            start = 0.0
        if end is None:
            end = duration

        if start < 0.0:
            start += duration
        if end < 0.0:
            end += duration

        if start < 0.0:
            raise IndexError("The slice start position ({} s) is out of "
                             "bounds. Filename: {}".format(start, fname))
        if end < 0.0:
            raise IndexError("The slice end position ({} s) is out of bounds "
                             "Filename: {}".format(end, fname))

        if start > end:
            raise IndexError("The slice start position ({} s) is later than "
                             "the slice end position ({} s)."
                             .format(start, end))

        if end > duration:
            raise AudioLengthError("The slice end time ({} s) is out of "
                                   "bounds (> {} s) Filename: {}"
                                   .format(end, duration, fname))

        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        data = sndfile.read(frames=end_frame - start_frame, dtype='float32')

        return cls(data, sample_rate)

    @classmethod
    def from_wav_data(cls, wav_data):
        """
        Create AudioSegment from wav data.

        Args:
            :param wav_data: binary wav data
            :type wav_data: basestring

        Returns:
            AudioSegment
        """
        # This scales the resulting float32 samples to [-1,+1] full-scale
        try:
            sndfile = soundfile.SoundFile(BytesIO(wav_data))
        except RuntimeError as exc:
            raise AudioFormatError('{}'.format(exc))

        if sndfile.format != 'WAV':
            raise AudioFormatError('Data is not wav data.')

        sample_rate = sndfile.samplerate
        samples = sndfile.read(dtype='float32')

        return cls(samples, sample_rate)

    def to_wav_file(self, output_filepath, dtype='int16'):
        """
        Save AudioSegment to disk as wav file.

        Args:
            :param output_filepath: path where wav file will be saved
            :type output_filepath: basestring
            :param dtype: data type of saved samples Default is 'int16'.
            :type dtype: basestring(numpy.dtype)

        Note:
            Saving an AudioSegment to 'int16' will result in a loss
            of precision since samples are stored as 'float32' internally.
        """
        samples = convert_samples_from_float32(self._samples, dtype)
        dtype = np.dtype(dtype)
        subtype_map = {
            np.int16: 'PCM_16',
            np.int32: 'PCM_32',
            np.float32: 'FLOAT',
            np.float64: 'DOUBLE',
        }

        soundfile.write(
            output_filepath,
            samples,
            self.sample_rate,
            format='WAV',
            subtype=subtype_map[dtype.type])

    @property
    def samples(self):
        """
        Return a copy of the audio samples as a numpy array.

        Returns:
            samples (1darray.float32)
        """
        return self._samples.copy()

    @property
    def sample_rate(self):
        """
        Return audio sample rate.
        """
        return self._sample_rate

    @property
    def num_samples(self):
        """
        Return number of audio samples per channel.
        """
        return self._samples.shape[0]

    @property
    def length_in_sec(self):
        """
        Return length in seconds.
        """
        return self._samples.shape[0] / self._sample_rate

    @property
    def rms_db(self):
        """
        Return root mean square energy of the audio in decibels (dB).

        Returns:
            float: rms in decibels wrt to signal with magnitude one.
        """
        if self._rms_db is None:
            # set RMS value
            mean_square = np.mean(self._samples**2)
            # sqrt => multiply by 10 instead of 20 for dBs
            self._rms_db = 10 * np.log10(mean_square)
        return self._rms_db

    def normalize(self, target_db=DEFAULT_NORMALIZATION, max_gain_db=300.0):
        """
        Normalize audio to desired RMS value in decibels.

        Args:
            :param target_db: Target RMS value in decibels.This value 
                should be less than 0.0 as 0.0 is full-scale audio.
            :type target_db: float, optional
            :param max_gain_db: Max amount of gain in dB that can be applied
                for normalization.  This is to prevent nans when attempting
                to normalize a signal consisting of all zeros.
            :type max_gain_db: float, optional

        Returns:
            AudioSegment: new AudioSegment with normalized samples.

        Raises:
            NormalizationWarning: if the required gain to normalize the
                segment to the target_db value exceeds max_gain_db.
        """
        gain_db = target_db - self.rms_db
        if gain_db > max_gain_db:
            warnings.warn(
                "Unable to normalize segment to {} dB because it has an RMS "
                "value of {} dB and the difference exceeds max_gain_db ({} dB)"
                .format(target_db, self.rms_db, max_gain_db),
                NormalizationWarning)
        gain_db = min(max_gain_db, target_db - self.rms_db)
        return self.apply_gain(gain_db)

    def normalize_online_bayesian(self,
                                  target_db,
                                  prior_db,
                                  prior_samples,
                                  startup_delay=0.0):
        """
        Normalize audio using a production-compatible online/causal algorithm.

        Note:
            This uses an exponential likelihood and gamma prior to make
            online estimates of the RMS even when there are very few samples.

        Args:
            :param target_db: Target RMS value in decibels
            :type target_bd: scalar
            :param prior_db: Prior RMS estimate in decibels
            :type prior_db: scalar
            :param prior_samples: Prior strength in number of samples
            :type prior_samples: scalar
            :param startup_delay: Default: 0.0 s. If provided, this
                function will accrue statistics for the first startup_delay
                seconds before applying online normalization.
            :type startup_delay: scalar

        Returns:
            SpeechDLSegment: New segment with normalized samples.
        """
        # Estimate total RMS online
        startup_sample_idx = min(self.num_samples - 1,
                                 int(self.sample_rate * startup_delay))
        prior_mean_squared = 10.**(prior_db / 10.)
        prior_sum_of_squares = prior_mean_squared * prior_samples
        cumsum_of_squares = np.cumsum(self.samples**2)
        sample_count = np.arange(len(self)) + 1
        if startup_sample_idx > 0:
            cumsum_of_squares[:startup_sample_idx] = \
                cumsum_of_squares[startup_sample_idx]
            sample_count[:startup_sample_idx] = \
                sample_count[startup_sample_idx]
        mean_squared_estimate = ((cumsum_of_squares + prior_sum_of_squares) /
                                 (sample_count + prior_samples))
        rms_estimate_db = 10 * np.log10(mean_squared_estimate)

        # Compute required time-varying gain
        gain_db = target_db - rms_estimate_db

        # Apply gain to new segment
        return self.apply_gain(gain_db)

    def normalize_ewma(self,
                       target_db,
                       decay_rate,
                       startup_delay,
                       rms_eps=1e-6,
                       max_gain_db=300.0):
        startup_sample_idx = min(self.num_samples - 1,
                                 int(self.sample_rate * startup_delay))
        mean_sq = self.samples**2
        if startup_sample_idx > 0:
            mean_sq[:startup_sample_idx] = \
                np.sum(mean_sq[:startup_sample_idx]) / startup_sample_idx
        idx_start = max(0, startup_sample_idx - 1)
        initial_condition = mean_sq[idx_start] * decay_rate
        mean_sq[idx_start:] = lfilter(
            [1.0 - decay_rate], [1.0, -decay_rate],
            mean_sq[idx_start:],
            axis=0,
            zi=[initial_condition])[0]
        rms_estimate_db = 10.0 * np.log10(mean_sq + rms_eps)
        gain_db = target_db - rms_estimate_db
        if np.any(gain_db > max_gain_db):
            warnings.warn(
                "Unable to normalize segment to {} dB because it has an RMS "
                "value of {} dB and the difference exceeds max_gain_db ({} dB)"
                .format(target_db, self.rms_db, max_gain_db),
                NormalizationWarning)
            gain_db = np.minimum(gain_db, max_gain_db)
        return self.apply_gain(gain_db)

    def apply_gain(self, gain_db):
        """
        Apply gain in decibels to new instance.

        Args:
            :param gain_db: Gain in decibels to apply to samples.
                If a 1darray, must be of same length as segment.
            :type gain_db: float, 1darray

        Returns:
            AudioSegment: new AudioSegment with gain applied.
        """
        linear_gain = db_to_amplitude(gain_db)
        # return newly constructed instance
        return type(self)(linear_gain * self._samples, self._sample_rate)

    def subsegment(self, start_sec=None, end_sec=None):
        """
        Return new AudioSegment containing audio between given boundaries.

        Args:
            :param start_sec: Beginning of subsegment in seconds,
                (beginning of segment if None).
            :type start_sec:  scalar
            :param end_sec: End of subsegment in seconds,
                (end of segment if None).
            :type end_sec: scalar

        Return:
            subsegment (AudioSegment): New AudioSegment containing specified
                subsegment.

        Note:
            See the .__getitem__() docstring for more specifics on how
                boundary values are handled.
        """
        # Default boundaries
        if start_sec is None:
            start_sec = 0.0
        if end_sec is None:
            end_sec = self.length_in_sec

        # negative boundaries are relative to end of segment
        if start_sec < 0.0:
            start_sec = self.length_in_sec + start_sec
        if end_sec < 0.0:
            end_sec = self.length_in_sec + end_sec

        start_sample = int(round(start_sec * self._sample_rate))
        end_sample = int(round(end_sec * self._sample_rate))
        samples = self._samples[start_sample:end_sample]

        return type(self)(samples, sample_rate=self._sample_rate)

    def resample(self, new_sample_rate, quality='sinc_medium'):
        """
        Resample audio and return new AudioSegment.

        This resamples the audio to a new sample rate and returns a brand
        new AudioSegment.  The existing AudioSegment is unchanged.

        Args:
            :param new_sample_rate: target sample rate
            :type new_sample_rate: scalar
            :param quality: One of {'sinc_fastest', 'sinc_medium', 'sinc_best'}.
                Sets resampling speed/quality tradeoff.
                See http://www.mega-nerd.com/SRC/api_misc.html#Converters
            :type quality: basestring

        Returns:
            AudioSegment: new AudioSegment with resampled audio.
        """
        resample_ratio = new_sample_rate / self._sample_rate
        new_samples = scikits.samplerate.resample(
            self._samples, r=resample_ratio, type=quality)
        return type(self)(new_samples, new_sample_rate)

    def speed_change(self, new_rate):
        """
        Change the speed of a sample through linear interpolation.

        Args:
            :param new_rate: rate of speed change, where 1.0 is unchanged.
                    Rates > 1.0 speed up the sample
                    Rates < 1.0 slow down the sample
            :type new_rate: float

        Returns:
            AudioSegment: new AudioSegment with modified audio.

        """
        samples = self._samples
        assert (new_rate > 0)
        assert (samples.dtype == np.float32)
        length_factor = 1.0 / new_rate
        length = samples.shape[0]
        new_length = int(length * length_factor)
        gt_indices = np.arange(length)
        new_indices = np.linspace(start=0, stop=length, num=new_length)
        new_samples = np.interp(new_indices, gt_indices, samples)
        return type(self)(new_samples, self._sample_rate)

    def pad_silence(self, duration, sides='both'):
        """
        Pads this audio sample with a period of silence.

        Args:
            :param duration: length of silence in seconds to pad
            :type duration: float
            :param sides:
                'beginning' - adds silence in the beginning
                'end' - adds silence in the end
                'both' - adds silence in both the beginning and the end.
            :type sides: basestring

        Returns:
            An audio segment with silence padded on the specified sides. If a
            zero duration is provided, self is returned.
        """
        if duration == 0.0:
            return self
        cls = type(self)
        silence = cls.make_silence(duration, self._sample_rate)
        if sides == "beginning":
            padded = cls.concatenate(silence, self)
        elif sides == "end":
            padded = cls.concatenate(self, silence)
        elif sides == "both":
            padded = cls.concatenate(silence, self, silence)
        else:
            raise ValueError("Unknown value for the kwarg 'sides'")
        return padded

    def convolve(self, ir, allow_resampling=False):
        """
        Convolve this audio segment with the given filter.

        Args:
            :param ir: impulse response
            :type ir: AudioSegment
            :param allow_resampling: indicates whether resampling is allowed
                when the ir has a different sample rate from this signal.
            :type allow_resampling: scalar

        Returns:
            convolved (AudioSegment): a new AudioSegment object containing
                the convolved signal.
        """
        if allow_resampling and self.sample_rate != ir.sample_rate:
            ir = ir.resample(self.sample_rate)

        if self.sample_rate != ir.sample_rate:
            raise SampleRateError("Impulse response sample rate ({}Hz) is "
                                  "equal to base signal sample rate ({}Hz)."
                                  .format(ir.sample_rate, self.sample_rate))

        samples = signal.fftconvolve(self.samples, ir.samples, "full")
        convolved = type(self)(samples, self.sample_rate)
        return convolved

    def convolve_and_normalize(self, ir, allow_resampling=False):
        """
        Convolve and normalize the resulting audio segment so that it
        has the same average power as the input signal.

        Args:
            :param ir: impulse response
            :type ir: AudioSegment
            :param allow_resampling: indicates whether resampling is allowed
                when the ir has a different sample rate from this signal.
            :type allow_resampling: scalar

        Returns:
            convolved (AudioSegment): a new AudioSegment object containing
                the convolved signal.
        """
        convolved = self.convolve(ir, allow_resampling=allow_resampling)
        convolved = convolved.normalize(target_db=self.rms_db)
        return convolved


class SpeechDLSegment(AudioSegment):
    """
    Subclassed AudioSegment with routines specific to Speech-DL
    preprocessing.
    """

    @classmethod
    def from_file(cls, filepath):
        """
        Create SpeechDLSegment from wav, pcm, or sequence file.

        Args:
            :param filepath: path to wav/pcm file or sequence file index. If
                wav/pcm filepath is prefixed with 'net:' netio will be used.
            :type filepath:  basestring

        Returns:
            SpeechDLSegment
        """
        # Note The sequence files meant to speed up the system by converting
        # random IO access to sequential access. This concept is irrelevant in
        # network I/O. The plan is to convert all seqfiles to .wav files
        # (and datasource json files accordingly) on I/O servers.

        ext = os.path.splitext(filepath)[1]
        over_net = filepath.startswith("net:")
        if over_net:
            data = net.pop_object(filepath)
            if ext == '.wav':
                return cls.from_wav_data(data)
            else:
                raise TypeError("Unsupported file extension: {}".format(ext))
        else:
            if ext == '.wav':
                return cls.from_wav_file(filepath)
            else:
                raise TypeError("Unsupported file extension: {}".format(ext))

    def random_subsegment(self, subsegment_length, rng=None):
        """
        Return a random subsegment of a specified length in seconds.

        Args:
            :param subsegment_length: Subsegment length in seconds.
            :type subsegment_length: scalar
            :param rng: Random number generator state
            :type rng: random.Random [optional]


        Returns:
            clip (SpeechDLSegment): New SpeechDLSegmen containing random
            subsegment of original segment.

        Raises:
            AudioLengthError: If subsegment_length is greater than original
                segment.
        """
        if rng is None:
            rng = random.Random()

        if subsegment_length > self.length_in_sec:
            raise AudioLengthError("Length of subsegment must not be greater "
                                   "than original segment.")
        start_time = rng.uniform(0.0, self.length_in_sec - subsegment_length)
        return self.subsegment(start_time, start_time + subsegment_length)

    def save_subsegments(self, prefix, outdir, duration):
        """
        Saves consecutive non-overlapping clips of this segment
        as separate wav files.

        Args:
            :param prefix: Filename prefix to use for segments. Filename
                format is "{prefix}.{starting_milliseconds}.wav"
            :type prefix: basestring
            :param outdir: Output directory for saved segments.
            :type outdir: basestring
            :param duration: Max length of each segment in seconds.
            :type duration: scalar

        Returns:
            A list of (filename, duration) tuples, with a tuple for
            each segemented file.

        Note:
            * Only complete subsegments of length `duration` are saved, so
                if the segment length isn't evenly divisible by `duration`,
                some portion of the end of the segment will not be saved into
                a subsegment file.
            * The `outdir` is created if it doesn't exist.
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        split_files = []
        start_times = np.arange(0.0, self.length_in_sec, duration)
        for start_time in start_times:
            subsegment = self[start_time:start_time + duration]
            if subsegment.length_in_sec < duration:
                continue
            basename = '{0}.{1}.wav'.format(prefix, int(start_time * MSEC))
            fname = os.path.join(outdir, basename)
            subsegment.to_wav_file(fname)
            split_files.append((basename, duration))
        return split_files

    def add_noise(self,
                  noise,
                  snr_dB,
                  allow_downsampling=False,
                  max_gain_db=300.0,
                  rng=None):
        """
        Adds the given noise segment at a specific signal-to-noise ratio.
        If the noise segment is longer than this segment, a random subsegment
        of matching length is sampled from it and used instead.

        Args:
            :param noise: Noise signal to add.
            :type noise: SpeechDLSegment
            :param snr_dB: Signal-to-Noise Ratio, in decibels.
            :type snr_dB: scalar
            :param allow_downsampling: whether to allow the noise signal
                to be downsampled to match the base signal sample rate.
            :type allow_downsampling: boolean
            :param max_gain_db: Maximum amount of gain to apply to noise
                signal before adding it in.  This is to prevent attempting
                to apply infinite gain to a zero signal.
            :type max_gain_db: scalar
            :param rng: Random number generator state.
            :type rng: random.Random

        Returns:
            SpeechDLSegment: signal with noise added.

        Raises:
            SampleRateError: If noise sample rate is smaller than signal
                sample rate.
            AudioLengthError: If noise signal is shorter than base signal.
        """
        if rng is None:
            rng = random.Random()

        if allow_downsampling and noise.sample_rate > self.sample_rate:
            noise = noise.resample(self.sample_rate)

        if noise.sample_rate != self.sample_rate:
            raise SampleRateError("Noise sample rate ({}Hz) is not equal to "
                                  "base signal sample rate ({}Hz)."
                                  .format(noise.sample_rate, self.sample_rate))
        if noise.length_in_sec < self.length_in_sec:
            raise AudioLengthError("Noise signal ({} sec) must be at "
                                   "least as long as base signal ({} sec)."
                                   .format(noise.length_in_sec,
                                           self.length_in_sec))
        noise_gain_db = self.rms_db - noise.rms_db - snr_dB
        noise_gain_db = min(max_gain_db, noise_gain_db)
        noise_subsegment = noise.random_subsegment(self.length_in_sec, rng=rng)
        output = self + noise_subsegment.apply_gain(noise_gain_db)

        return output


def clip_samples(samples, dtype):
    """Clip samples for the range of the given dtype.

    Args:
        :param samples: input samples to clip
        :type samples: ndarray
        :param dtype: dtype to clip samples for
        :type dtype: basestring, numpy.dtype

    Returns:
        clipped_samples (ndarray): copy of input samples where values
            exceeding dtype range are clipped.

    Raises:
        Issues a ClippingWarning if any samples are clipped
    """
    dtype = np.dtype(dtype)

    output_samples = samples.copy()

    if 'int' in dtype.name:
        type_info_func = np.iinfo
    elif 'float' in dtype.name:
        type_info_func = np.finfo
    else:
        raise TypeError("Clipping not supported for type: {}".format(dtype))

    max_val = type_info_func(dtype).max
    min_val = type_info_func(dtype).min

    max_clipped = output_samples > max_val
    min_clipped = output_samples < min_val
    num_clipped = max_clipped.sum() + min_clipped.sum()
    output_samples[max_clipped] = max_val
    output_samples[min_clipped] = min_val

    if num_clipped > 0:
        warnings.warn("Clipped {} out of {} values."
                      .format(num_clipped, samples.size), ClippingWarning)

    return output_samples


def convert_samples_to_float32(samples):
    """
    Return float32 samples scaled such that full-scale range
    is [-FULL_SCALE, +FULL_SCALE]

    Args:
        :param samples: input samples of type int16 or float32/64
        :type samples: ndarray

    Returns:
        scaled_samples (ndarray.float32)
    """
    float32_samples = samples.astype('float32')  # always returns a copy

    # rescale integer inputs to full-scale
    if samples.dtype in np.sctypes['int']:
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= (FULL_SCALE / 2**(bits - 1))
    elif samples.dtype in np.sctypes['float']:
        pass
    else:
        raise TypeError("Unsupported wav sample type: {}".format(samples.dtype))

    return float32_samples


def convert_samples_from_float32(samples, dtype):
    """
    Return samples converted from float32 with full-scale range
    [-FULL_SCALE, +FULL_SCALE] to desired dtype.

    Note:
        Depending on the target dtype, the samples will be rescaled
        differently. For integer types, this involves rescaling full-scale
        float32 data to the maximum range supported by the integer type.
    Args:
        :param samples: input samples of type float32
        :type param: ndarray.float32
        :param dtype: output dtype
        :type dtype: str, numpy.dtype

    Returns:
        scaled_samples (ndarray.dtype): output samples of desired dtype.
    """
    dtype = np.dtype(dtype)  # convert possible str to numpy.dtype

    if samples.dtype != 'float32':
        raise TypeError("Input samples must be of type 'float32'")

    output_samples = samples.copy()

    if dtype.type in np.sctypes['int']:  # in list of signed integer types
        bits = np.iinfo(dtype).bits
        # rescale from full-scale to integer range
        output_samples *= (2**(bits - 1) / FULL_SCALE)
        output_samples = clip_samples(output_samples, dtype)
    elif dtype in np.sctypes['float']:  # in list of floating point types
        output_samples = clip_samples(output_samples, dtype)
    else:
        raise TypeError("Unsupported wav sample type: {}".format(dtype))

    return output_samples.astype(dtype)


def db_to_amplitude(db_val):
    """Return linear gain corresponding to relative decibel value.

    Args:
        :param db_val: Relative decibel value.
        :type db_val: float

    Returns:
        float: Linear gain corresponding to input decibels.
    """
    return 10.**(db_val / 20.)
