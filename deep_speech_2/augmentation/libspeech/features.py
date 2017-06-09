"""
Functions for computing spectrograms.

Here be functions for computing various types of spectrograms.

The primary function that should be called from outside this module
is `compute_specgram()`, which takes arguments that define the particular
type of spectrogram to be extracted.

Currently supported values for `spec_type` are:
    {'linear', 'cqt', 'pcqt', 'mel', 'multi', 'waveform'}

The configuration parameters specific to each `spec_type` are shown below.

'linear' (See `compute_linear_specgram()` below)
    ['window', 'max_freq']

'cqt' (See `compute_cqt()` below)
    ['fmin', 'fmax', 'n_bins']

'pcqt' (See `compute_pseudo_cqt()` below)
    ['window', 'fmin', 'fmax', 'n_bins']

'mel' (See `compute_mel_specgram()` below)
    ['window', 'fmin', 'fmax', 'n_bins']

'multi' (See `compute_multi_specgram()` below)
    ['spec_list' with elements {'spec_type', 'spec_params'}]

'waveform' (See `get_waveform()` below)
    []


Spectrograms are then postprocessed according to `post_spec_params`.
If `post_spec_params` is not provided in the experiment json file, the default
behavior is to apply log compression to all `spec_type`s except for `multi`
and `waveform`:
{
    "method": "log",
    "params": {}
}

Another postprocessing method is PCEN (https://arxiv.org/pdf/1607.05666v1.pdf).
An example is provided below, but for more detail, see the docstring of
`apply_pcen`.
{
    "method": "pcen",
    "params": {
        "alpha": 0.98,
        "delta": 2.0,
        "r": 0.5,
        "s": [0.015, 0.08]
    }
}
"""
from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal.signaltools import lfilter

EPS = 1e-16

SPEC_TYPES = {'linear', 'cqt', 'pcqt', 'mel', 'multi', 'waveform'}


def specgram_real(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Computes the spectrogram for a real signal.

    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.  Seems somewhat arbitrary thus
    # TODO, awni, check on the validity of this scaling.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def get_feat_dim(spec_type, **spec_params):
    """Return feature dimension of spectrogram.

    This returns the number of frequency bins present at each time step.

    Args:
        spec_type (str): Type of spectrogram to extract.
        spec_params (kwargs): dict of params specific to `spec_type.

    Returns:
        int: Feature dimension
    """
    if spec_type not in SPEC_TYPES:
        raise ValueError("Unknown spec_type `{}`. "
                         "Supported values: {}".format(spec_type, SPEC_TYPES))

    if spec_type == 'linear':
        return int(0.001 * spec_params['window'] * spec_params['max_freq']) + 1
    elif spec_type == 'multi':
        return sum(
            get_feat_dim(spec_config['spec_type'], **spec_config['spec_params'])
            for spec_config in spec_params['spec_list'])
    elif spec_type in {'cqt', 'pcqt', 'mel'}:
        return spec_params['n_bins']
    elif spec_type == 'waveform':
        return 1


def compute_volume_normalized_feature(audio_segment, feature_info):
    """Compute time x frequency feature matrix after normalizing the audio.
    We *do* convert to log-space.

    This is currently only used by web_api/api_server.py

    Args:
        audio_segment (SpeechDLSegment): input audio as SpechDLSegment.
        feature_info (dict): contains key-value pairs used to configure
            the feature extraction.  Required keys:
            {step, spec_type, spec_params}

    Returns:
        spec (float32 2darray): time x frequency spectrogram matrix
    """
    # old int16-based pydub AudioSegment was normalized to 70db so we'll
    # stick with that convention
    normalized_audio_segment = audio_segment.normalize(target_db=70)

    spec, _ = compute_specgram(
        normalized_audio_segment,
        step=feature_info['step'],
        spec_type=feature_info['spec_type'],
        **feature_info['spec_params'])

    assert spec.dtype == 'float32'

    # transpose to time x frequency
    return spec.T


def compute_specgram(audio_segment,
                     step=10,
                     shift=0,
                     spec_type='linear',
                     **spec_params):
    """Compute spectrogram from audio segment

    Note: The spectrogram is log-scaled unless `spec_type` is 'waveform'.

    Args:
        audio_segment (AudioSegment): Input audio segment
        step (scalar, optional): step size in ms between windows
        shift (scalar, optional): time in ms to shift the audio samples.
            Positive values are a time advance, negative values are a time
            delay.  This is done to allow for variable intra-window
            alignment with the spectrogram windows.  (So we don't overfit
            on a constant window alignment.)
        spec_type (str, optional): spectrogram type
        spec_params (kwargs, optional): Additional keyword args to be passed to
            specific spectrogram function.

    Returns:
        spec (float32 2darray): spectrogram as [freq, time]
        freqs (1d array): frequency corresponding to each fft bin
    """
    return compute_specgram_from_samples(
        audio_segment.samples,
        audio_segment.sample_rate,
        step=step,
        shift=shift,
        spec_type=spec_type,
        **spec_params)


def compute_specgram_from_samples(samples,
                                  sample_rate,
                                  step=10,
                                  shift=0,
                                  spec_type='linear',
                                  **spec_params):
    """Compute spectrogram from audio samples

    Note: The spectrogram is log-scaled unless `spec_type` is 'waveform'.

    Args:
        samples (1darray): input audio samples
        sample_rate (scalar): audio sample rate
        step (scalar, optional): step size in ms between windows
        shift (scalar, optional): time in ms to shift the audio samples.
            Positive values are a time advance, negative values are a time
            delay.  This is done to allow for variable intra-window
            alignment with the spectrogram windows.  (So we don't overfit
            on a constant window alignment.)
        spec_type (str, optional): spectrogram type
        spec_params (kwargs, optional): Additional keyword args to be passed to
            specific spectrogram function.

    Returns:
        spec (float32 2darray): spectrogram as [freq, time]
        freqs (1d array): frequency corresponding to each fft bin
    """
    shift_samples = int(sample_rate * shift / 1000)
    if shift_samples < 0:  # Time delay
        # Remove `shift_samples` samples from the end and prepend
        # `shift_samples` zeros to the beginning.
        samples[-shift_samples:] = samples[:shift_samples]
        samples[:-shift_samples] = 0
    elif shift_samples > 0:  # Time advance
        # Remove `shift_samples` samples from the beginning
        # and append `shift_samples` zeros to the end.
        samples[:-shift_samples] = samples[shift_samples:]
        samples[-shift_samples:] = 0

    if spec_type not in SPEC_TYPES:
        raise ValueError("Unknown spec_type `{}`. "
                         "Supported values: {}".format(spec_type, SPEC_TYPES))

    # Compute specific spectrogram type.
    if spec_type == 'linear':
        spec, freqs = compute_linear_specgram(
            samples, sample_rate, step=step, **spec_params)
    elif spec_type == 'multi':
        spec, freqs = compute_multi_specgram(
            samples, sample_rate, step=step, **spec_params)
    elif spec_type == 'cqt':
        spec, freqs = compute_cqt(
            samples, sample_rate, step=step, **spec_params)

    elif spec_type == 'pcqt':
        spec, freqs = compute_pseudo_cqt(
            samples, sample_rate, step=step, **spec_params)
    elif spec_type == 'mel':
        spec, freqs = compute_mel_specgram(
            samples, sample_rate, step=step, **spec_params)
    elif spec_type == 'waveform':
        spec, freqs = get_waveform(samples)

    # Convert samples to float32
    spec = spec.astype('float32')

    return spec, freqs


def postprocess_spectrogram(feats, method, params):
    """
    Applies various postprocessing steps to spectrograms.

    Args:
        method (str): Postprocessing method. Currently supported are
            'log' and 'pcen'.
        params (dict): Keyword arguments for the selected postprocessing
            `method`.

    Returns:
        Postprocessed feature matrix [freq, time]
    """
    if method == 'log':
        return np.log(feats + EPS)
    elif method == 'pcen':
        return apply_pcen(feats, **params)
    else:
        raise ValueError('Unknown postprocessing method "{}"'.format(method))


def apply_pcen(feats, alpha=0.98, delta=2.0, r=0.5, s=[0.025], epsilon=1e-6):
    """
    This performs per-channel energy normalization as described in

    https://arxiv.org/pdf/1607.05666v1.pdf

    Namely, it computes

    PCEN(t, f) = (E(t, f) / (\epsilon + M(t, f)) ^ \alpha + \delta) ^ r -
                 \delta ^ r,                                             (1)

    where M(t, f) = 1 / N \sum_i M_i(t, f) and

    M_i(t, f) = (1 - s_i) M_i(t - 1, f) + s_i E(t, f).                   (2)

    Args:
        feats (2d array): spectrogram as [freq, time]
        alpha (float): \alpha in (1)
        delta (float): \delta in (1)
        r (float): r in (1)
        s (list): [s_i for each i] in (2)
        epsilon (float): \epsilon in (1)

    Returns:
        Per-channel normalized feature matrix [freq, time]
    """
    assert np.all(feats >= 0.0)
    # Compute M
    M = np.zeros(feats.shape)
    for smoothing_coeff in s:
        M += lfilter(
            [smoothing_coeff], [1.0, -1.0 + smoothing_coeff], feats, axis=1)
    # For now, we take a simple average.
    M *= (1.0 / len(s))
    # PCEN
    feats_pcen = (feats / (M + epsilon)**alpha + delta)**r - delta**r
    return feats_pcen


def compute_linear_specgram(samples,
                            sample_rate,
                            step=10,
                            window=20,
                            max_freq=None):
    """Compute spectrogram from FFT energy.

    Args:
        samples (1darray): audio samples
        sample_rate (scalar): sample rate of audio samples
        step (scalar, default=10): step size in ms between windows
        window (scalar, default=20): FFT window size in ms
        max_freq (scalar, default=sample_rate/2): Only FFT bins corresponding
            to frequencies between  [0, max_freq] are returned.

    Returns:
        spec (2d array): spectrogram as [freq, time]
        freqs (1d array): frequency corresponding to each fft bin
    """
    if max_freq is None:
        max_freq = sample_rate / 2

    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half sample rate")

    if step > window:
        raise ValueError("Step size must not be greater than window size")

    hop_length = int(0.001 * step * sample_rate)
    fft_length = int(0.001 * window * sample_rate)
    pxx, freqs = specgram_real(
        samples,
        fft_length=fft_length,
        sample_rate=sample_rate,
        hop_length=hop_length)
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    return pxx[:ind, :], freqs[:ind]


def compute_multi_specgram(samples, sample_rate, step, spec_list):
    """Compute spectrogram by stacking the energy of multiple FFTs.

    Args:
        samples (1d array): audio samples
        sample_rate (scalar): sample rate of audio samples
        step (scalar, default=10): step size in ms between windows
        spec_list (list): list of dicts (one for each specgram) where each
            dict contains:
                * spec_type (str): spec_type of specgram
                * spec_params (dict): defines parameters specific to
                    `spec_type`.
    Returns:
        spec (2d array): spectrogram as [freq, time] where spectrograms
            from multiple FFTs are stacked along the `freq` axis.
        freqs (1d array): frequency corresponding to each location along
            the `freq` axis.  Note that with more than 1 FFT, this will not be
            monotonically increasing.
    """
    multi_specs = []
    multi_freqs = []
    for spec_config in spec_list:
        spec_type = spec_config['spec_type']
        spec_params = spec_config['spec_params']

        if spec_type == 'linear':
            # Pad with fft_length//2 samples so that center of each window
            # is aligned across FFTs.
            # The other spec_types already do this internally.
            fft_length = int(0.001 * spec_params['window'] * sample_rate)
            pad_length = fft_length // 2
            padded_samples = np.pad(samples, pad_length, mode='reflect')
        else:
            padded_samples = samples

        spec, freqs = compute_specgram_from_samples(
            padded_samples,
            sample_rate,
            step=step,
            spec_type=spec_type,
            **spec_params)

        multi_specs.append(spec)
        multi_freqs.append(freqs)

    # truncate all specgrams to length of shortest specgram
    min_frames = min(sp.shape[1] for sp in multi_specs)
    multi_specs = [sp[:, :min_frames] for sp in multi_specs]

    spec = np.concatenate(multi_specs, axis=0)
    freqs = np.concatenate(multi_freqs, axis=0)

    return spec, freqs


def compute_mel_specgram(samples,
                         sample_rate,
                         step=10,
                         window=20,
                         fmin=80,
                         fmax=None,
                         n_bins=161):
    """Compute mel-band spectrogram from FFT energy.

    The bands of this spectrogram are distrubted uniformly on the mel scale.

    Args:
        samples (1d array): audio samples
        sample_rate (scalar): sample rate of audio samples
        step (scalar, default=10): step size in ms between windows
        window (scalar, default=20): FFT window size in ms
        fmin (scalar, optional): center frequency of lowest mel band
        fmax (scalar, default=sample_rate/2): upper bound on center frequency
            of highest mel band.
        n_bins (scalar, optional): number of mel bins

    Returns:
        spec (2d array): spectrogram as [freq, time]
        freqs (1d array): frequency corresponding to each mel bin
    """
    import librosa

    if fmax is None:
        fmax = sample_rate / 2

    if fmax > sample_rate / 2:
        raise ValueError("fmax must not be greater than half sample rate")

    if step > window:
        raise ValueError("Step size must not be greater than window size")

    fft_length = int(window / 1000 * sample_rate)
    hop_length = int(step / 1000 * sample_rate)

    mel_spec = librosa.feature.melspectrogram(
        samples,
        sr=sample_rate,
        n_fft=fft_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        n_mels=n_bins)
    freqs = librosa.mel_frequencies(n_mels=n_bins, fmin=fmin, fmax=fmax)

    return mel_spec, freqs


def compute_cqt(samples, sample_rate, step=10, fmin=80, fmax=None, n_bins=161):
    """Compute Constant-Q Transform (CQT)

    This CQT produces a spectrogram containing `n_bins` at each time
    step where the bins are distributed logarithmically between `fmin`
    and `fmax`.  `fmin` is the actual center frequency of the lowest frequency
    bin, while `fmax` is an upper bound on the center frequency of the
    highest frequency bin.

    This uses the `hybrid_cqt` routine from the librosa library.
    http://bmcfee.github.io/librosa/generated/librosa.core.hybrid_cqt.html

    For more on the CQT, see:
        https://en.wikipedia.org/wiki/Constant_Q_transform

    Args:
        samples (1d array): audio samples
        sample_rate (scalar): sample rate of audio samples
        step (scalar, optional): step size in ms between windows
        fmin (scalar, optional): center frequency of lowest CQT filter
        fmax (scalar, default=sample_rate/2): upper bound on center frequency
            of highest CQT filter.
        n_bins (scalar, optional): number of CQT bins

    Returns:
        spec (2d array): spectrogram as [freq, time]
        freqs (1d array): frequency corresponding to each fft bin

    Note:
        There must be an integer number of bins in each octave, hence the
        fact that `fmax` is an upper bound and not the actual center
        frequency of the highest frequency bin.
    """
    import librosa

    if fmax is None:
        fmax = sample_rate / 2

    if fmax > sample_rate / 2:
        raise ValueError("fmax must not be greater than half sample rate")

    num_octaves = np.log2(fmax / fmin)
    bins_per_octave = int(n_bins / num_octaves) + 1
    hop_length = int(0.001 * step * sample_rate)

    freqs = librosa.cqt_frequencies(
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave, )

    spec = librosa.hybrid_cqt(
        samples,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        tuning=0.0)

    return spec, freqs


def compute_pseudo_cqt(samples,
                       sample_rate,
                       step=10,
                       window=20,
                       fmin=80,
                       fmax=None,
                       n_bins=161):
    """Compute Pseudo Constant-Q Transform (PCQT)

    The PCQT produces a spectrogram containing `n_bins` at each time
    step where the bins are distributed logarithmically between `fmin`
    and `fmax`.  `fmin` is the approximate center frequency of the
    lowest frequency bin, while `fmax` is an upper bound on the
    center frequency of the highest frequency bin.

    Unlike the CQT, the PCQT only uses a single FFT size, so it
    doesn't achieve the actual time/frequency-resolution tradeoff achieved
    by the CQT.  Because of this, at small FFT window sizes, there will be
    smearing across frequency in low frequency bins, and at large
    window sizes, there will be smearing across time.

    This uses the `pseudo_cqt` routine from the librosa library.
    http://bmcfee.github.io/librosa/generated/librosa.core.pseudo_cqt.html

    For more on the CQT, see:
        https://en.wikipedia.org/wiki/Constant_Q_transform

    Args:
        samples (1d array): audio samples
        sample_rate (scalar): sample rate of audio samples
        step (scalar, optional): step size in ms between windows
        window (scalar, default=20): FFT window size in ms
        fmin (scalar, optional): center frequency of lowest CQT filter
        fmax (scalar, default=sample_rate/2): upper bound on center frequency
            of highest CQT filter.
        n_bins (scalar, optional): number of CQT bins

    Returns:
        spec (2d array): spectrogram as [freq, time]
        freqs (1d array): frequency corresponding to each fft bin

    Note:
        There must be an integer number of bins in each octave, hence the
        fact that `fmax` is an upper bound and not the actual center
        frequency of the highest frequency bin.
    """
    import librosa

    if fmax is None:
        fmax = sample_rate / 2

    if fmax > sample_rate / 2:
        raise ValueError("fmax must not be greater than half sample rate")

    num_octaves = np.log2(fmax / fmin)
    bins_per_octave = int(n_bins / num_octaves) + 1
    hop_length = int(0.001 * step * sample_rate)

    freqs = librosa.cqt_frequencies(
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave, )

    spec = librosa.pseudo_cqt(
        samples,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        tuning=0.0)

    return spec, freqs


def get_waveform(samples):
    """Get raw waveforms

    Args:
        samples (1d array): audio samples

    Returns:
        spec (float32 2darray): original samples with leading singleton
            dimension added.
        freqs (1d array): a 1d array with only one element 'None'
    """
    spec = samples.astype('float32')
    spec = spec[np.newaxis, :]
    freqs = np.array([None])

    return spec, freqs
