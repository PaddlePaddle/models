#data\_config

## server
*true | false*

depending on whether you want to use the data-server or not
        If the data-server is not used, make sure that the data\_dir points to
        a path which has the preprocessed data in the correct format

## force\_server
*true | false*

If you want evaluation routines to generate features on the go, set this to
        true. Do **not** use for training routines, the training dataset
        will be lost or rewritten by evaluation datasets.

## max\_duration
*float, (default=15)*

Require all utterances to be no greater than this number of seconds.

## min\_duration
*float, (default=0)*

Require all utterances to be at least this many seconds.

## starting\epoch
*int, (default=0)*

Epoch to start the run at.

## starting\_epoch\_iteration
*int, (default=0)*

Iteration within the epoch the start at. If the value is larger than the number of iterations within the epoch, the next epoch is immediately started.

## save\_dir
*path*

The location where the preprocessed data is saved for consumption by RNN
        Slurm creates a folder /tmp/$USER during job initialization and removes
        this folder when the job is canceled/completes. The default configs now
        all use this folder so that the tmp dir needn’t be cleared each time.
        If you want to save preprocess state, use a different path

## train\_set\_sample\_rate
 If you want to use a random subset of the specified training data, instead
        of all of it, set this field equal to the fraction of the data you want
        to use.  If you don't specify this field, or specify 1.0, all of the data
        will be used

## cross\_validation
A list of cross validation data sets to be tested every epoch.  Each will be written
out to its own log file.

### named\_dev\_sets
*dev | dev\_spk | dev\_noise | dev\_train\_holdout | dev\_train\_sample*

If you are using WSJSet, or mark one of the data\_sources to be split,
        you can do cross-validation on the specified set(s). Note that this requires
        you to use either WSJ or pass the dev\_set\_sample\_rate and/or
        num\_speakers\_in\_hold\_out parameters.  You can test on multiple dev sets, but
        only the first one in the list will be used for saving and early stopping.
        If you wish to use dev\_train\_holdout and/or dev\_train\_sample, you will need
        to specify the split\_train option, and pass the num\_train\_holdout and
        num\_train\_sample options respectively. These options are set per data source.

### other\_dev\_sets
*paths*

In addition to the named dev sets above (which are derived from training data),
        you can also test on any desired additional data sets every epoch by giving a list
        of paths to their data directories.

##data\_sources

### location path
This is either the directory where all the data is present, or some
        summary about where all the data is present. Almost all data should be in
    `/local/data/`. The current English sequence JSON summary file is `/local/data/english_complete_seq_20150808/complete.json`.
    The previous English sequence JSON summary file is `/local/data/complete_seq/complete.json`. The current Chinese
    sequence JSON summary file is `/local/data/chinese_complete_seq/complete_chinese_seq.json`.

### prev\_iter path
When more data is added to a dataset, we want to retain the same
        hold out as from the previous iteration. When that is the case,
        provide this param. Look for <DataSet>.pkl file in this directory

### split
*true | false*

This optional value defaults to false. When set to true, it carves out a
        set of files from the data-sources for cross-validation. Use the
        `prev_iter` path to retain the same carved out set as before, if your
        dataset expands (or changes)

### train\_only
*true | false*

Defaults to false. When using JSONDataset, the dev sets maybe hardcoded.
        If you do not want to use the dev set from a particular dataset, set
        `train_only` to true

### add\_noise
*true | false*

Defaults to true. When set to false, noise is not added to this dataset.

### jitter
*true | false*

If true, randomly shifts the signal by one half the step size,
        either left or right. If false, no random shifting is performed.
        Jitter defaults to true if not specified.

### num\_train\_sample
*int (default:0)*
Number of utterances to be kept in dev\_train\_sample dataset.
        This set is used to measure how well a model is fitting the training
    data. As a "dev" set, no noise / augmentation and other regularization
    will be applied on evaluation.

### split_train
*true | false*
Defaults to false. If true, enables splitting of training and unclassified
        data into separate train and dev\_train\_holdout datasets.

### num\_train\_holdout
*int (default:0)*
Number of utterances to be kept in dev\_train\_holdout dataset. This will be
        used to compare performance over datasets across models. As a "dev"
    set, no noise / augmentation and other regularization will be applied
    on evaluation. It will be used only if split\_train is true.

## use\_global\_normalization
*true | false*

Defaults to true, if no other normalization methods are selected.

## use\_online\_normalization
*true | false*

Defaults to false. When set to true, `online_normalization_parameters` must
        be specified.

## online\_normalization\_parameters
*dict*

### prior\_db
*scalar*
    Prior RMS estimate in decibels.

### prior\_samples
*scalar*
    Prior strength in number of samples.

### startup\_delay
*scalar*
    Optional argument. The first startup_delay seconds of audio is used to
    estimate initial audio volume.

## use\_ewma\_normalization
*true | false*

Defaults to false. When set to true, `ewma_normalization_parameters` must
        be specified, and audio volume used for normalization is estimated by
        exponentially weighted moving average.

## ewma\_normalization\_parameters
*dict*

### decay\_rate
*scalar*
    estimate_n := decay_rate * estimate_{n - 1} + (1 - decay_rate) * sample_n^2

### rms\_eps
*scalar*
    a small value added before applying log to prevent negative infinity

### startup\_delay
    The first startup_delay seconds of audio is used to estimate initial audio volume.

## create\_identity\_transform\_stats
*true | false*
Defaults to false. If set, input features are not normalized.
        e.g. feature mean and standard deviation are set to 0 and 1, respectively.

## augmentation\_pipeline

A configurable pipeline specified in the json file that can chain an
    arbitrary number of augmentation modules. Each module must specify
    its `type` and `rate`. An example usage is provided in
    minimal\_wsj.json. The rest of the section details all available
    modules.

    To not perform data augmentation, leave this section blank
    (e.g. an empty list [])

### type
*string {'walla_noise', 'impulse_response', 'online_bayesian_normalization', 'resampler', 'volume_change'}*

Augmentation block type.

### rate
*float*

Probability of applying a particular module.

### walla\_noise

Adds noise to input audio segment.

#### source
*string {'turk', 'freesound', 'chime'}*

Source from which noises were downloaded. This field is mandatory only for
    the "turk" noises.

#### noise\_dir
*path*

This is the directory from which noises are loaded.

#### index\_file
*path*

A file containing a list of noise files under noise\_dir. The default value
    for the "turk" source is "`noise\_dir`/noise-samples-list". For all
    the other soures, the default value is
    "`noise\_dir`/audio\_index\_commercial.txt".

#### snr\_min, snr\_max
*float*

Noise for each utterance is scaled so as to have an snr randomly
    selected from the interval `[snr_min, snr_max]`.  This parameter must
    be specified if `noise_dir` is specified.

#### allow\_downsampling
*bool*

Whether to allow implicit downsampling of the noise so it matches the sample rate
    of the input audio.  If this is disbabled and the noise sample rate does not
    match that of the input audio, an error will be raised.

#### tags
*list*

If the noise index file contains tags for each noise, providing this list
    enables the module to draw noises only with those tags.

#### tag\_distr
*dict*

A dictionary mapping from a tag to a probability mass (probability masses are
    normalized automatically). This specifies the desired noise distribution
    in the augmented training set. This feature works only if the noise tags
    are provided in the index file.

### impulse\_response

Convolves the input audio segment with an impulse response. Note that the output
    may no longer be normalized after this operation; use
    online\_bayesian\_normalization.

#### ir\_dir
*path*

This is the directory from which impulse responses are loaded.

#### index\_file
*path*

A file containing a list of impulse response files under ir\_dir.

#### tags
*list*

If the ir index file contains tags for each response, providing this list
    enables the module to draw ir's only with those tags.

#### tag\_distr
*dict*

A dictionary mapping from a tag to a probability mass (probability masses are
    normalized automatically). This specifies the desired impulse response
    distribution in the augmented training set. This feature works only if the
    IR tags are provided in the index file.

### online\_bayesian\_normalization

#### target\_db
*scalar*
    Target RMS value in decibels.

#### prior\_db
*scalar*
    Prior RMS estimate in decibels.

#### prior\_samples
*scalar*
    Prior strength in number of samples.

### resampler

Resamples audio.

#### new\_sample\_rate
*scalar*

New sample rate in Hz.

### volume\_change

Used for multi-loudness training.

#### min\_gain\_dBFS
*scalar*

Minimum gain in dBFS.

#### max\_gain\_dBFS
*scalar*

Maximum gain in dBFS.

## char\_map
*string*

Specify path to a particular char-map. If not specified, it generates a
        char-map for the datasets

## step
*float*

Spectrogram "hop size" in milliseconds.

## sample_rate
*int/float*

Input audio must have this sample rate or an error will be raised.

## spec\_type
*string {'linear', 'multi', 'cqt', 'pcqt', 'mel', 'waveform'}*

Type of spectrogram to extract.  `linear` is default.

Other types are `cqt` (Constant-Q Transform), `pcqt` (Pseudo Constrant-Q Transform), `mel` (mel-band spectrogram), `multi` which stacks multiple spectrograms along the frequency axis, and `waveform` (raw signals without any transformation).

## spec_params
*dict*

Parameters that are specific to each `spec_type` above.

For `linear`:
* `window`: The FFT window size in milliseconds
* `max_freq`: Only FFT bins corresponding to frequencies between  `[0, max_freq]` are returned.
```json
"spec_type": "linear",
"spec_params": {
    "window": 20,
    "max_freq": 8000
}
```

For `cqt`:
* `fmin`: Center frequency of lowest CQT filter
* `fmax`: Upper bound on center frequency of highest CQT filter.
* `n_bins`: number of CQT bins
```json
"spec_type": "cqt",
"spec_params": {
    "fmin": 80,
    "fmax": 8000,
    "n_bins": 161
}
```

For `pcqt`:
* `window`: The FFT window size in milliseconds
* `fmin`: Center frequency of lowest bin
* `fmax`: Upper bound on center frequency of highest bin
* `n_bins`: number of CQT bins
```json
"spec_type": "pcqt",
"spec_params": {
    "window": 20,
    "fmin": 80,
    "fmax": 8000,
    "n_bins": 161
}
```

For `mel`:
* `window`: The FFT window size in milliseconds
* `fmin`: Center frequency of lowest bin
* `fmax`: Upper bound on center frequency of highest bin
* `n_bins`: number of CQT bins
```json
"spec_type": "mel",
"spec_params": {
    "window": 20,
    "fmin": 80,
    "fmax": 8000,
    "n_bins": 161
}
```

For `multi`:
* `param_list`: list of JSON objects (dicts) where each dict defines the following params:
  * `window`: The FFT window size in milliseconds
  * `max_freq`: Only FFT bins corresponding to frequencies between  `[0, max_freq]` are returned.
```json
"spec_type": "mult",
"spec_params": {
    "spec_list": [
        {
            "spec_type": "linear",
            "spec_params": {
                "window": 20,
                "max_freq": 8000
            }
        },
        {
            "spec_type": "linear",
            "spec_params": {
                "window": 40,
                "max_freq": 8000
            }
        }
    ]
}
```

For `waveform`:
* No params.
```json
"spec_type": "waveform",
"spec_params": {
}
```
## spec_parallelism, (default=1)
*int*

Preprocess currently does not use multiproccessing to calculate spectrograms in parallel.
For linear spectrogram there is no impact on performance, but when calculating more complicated
spectrograms, set this to a positive value. This parameter indicates the size of the process
pool used by the preprocess script (per data-parallel rank). For CQT set to 12.

## post\_spec\_params
*list | dict*

A configurable pipeline expressed as a list of dictionaries specifying how spectrograms are postprocessed.
Postprocessing methods are chained and each entry is specified as follows:

```
{
    "method": "some_postprocessing_method",
    "params": {
        "parameter1": 1.0,
        ...
    }
}
```

Currently supported are

```
'''
Log compression. If `post_spec_params` is not specified, this option is
enabled by default for all `spec_types` other than `waveform` and `multi`.
'''
{
    "method": "log",
    "params": {}
}
```

```
'''
Per-channel energy normalization (https://arxiv.org/pdf/1607.05666v1.pdf).

PCEN(t, f) = (E(t, f) / (\epsilon + M(t, f)) ^ \alpha + \delta) ^ r -
             \delta ^ r,                                             (1)

where M(t, f) = 1 / N \sum_i M_i(t, f) and

M_i(t, f) = (1 - s_i) M_i(t - 1, f) + s_i E(t, f).                   (2)
'''
{
    "method": "pcen",
    "params": {
        "alpha": \alpha in (1),
        "delta": \delta in (1),
        "r": r in (1),
        "s": [s_i for each i] in (2)
    }
}
```

## scheduler\_config

A description of the type of scheduler to be used for the training pipeline.

### class
*string, default=schedulers.sortagrad\_scheduler.SortagradScheduler*
This is the path to the scheduler relative to the libspeech directory. It is used to initialize
     the required scheduler for the training pipeline. Look in the schedulers directory to see
     other implemented schedulers. Their pydocs will include required options.

### <other options>
These are other options relevant to the selected scheduler. BaseScheduler does not have any other
      options.
