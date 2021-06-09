# Audio preprocessing using PaddleAudio

This example demonstrates how to use paddleaudio for feature extraction in the preprocessing stage.

## Introduction

Training deep neural networks for audio usually requires heavy computations in the feature extraction stage. Sometimes these computations are repeated between epochs. Therefore, preprocessing the audio data into fbank features before training is beneficial, as it will save a lot of time and most importantly, it saves a lot of energy.

## Quick start

We provide a script file along with sample config to do the feature extraction and feature storage using hdf5 format. Users can get started by running

``` sh
python wav2fbank.py -c sample_config.yaml
```

, and the config file is self-explained and listed as follows,

``` yaml
#wav2fbank config file
fbank:
  sr: 16000 # sample rate
  window_size: 400 #25ms
  hop_length: 160 #10ms
  n_mels: 40
  fmin: 20
  fmax: 7600
  to_db: False # set true to compute log FB
  window: hann
  center: True
  pad_mode: reflect
  ref: 1.0
  amin: !!float 1e-10
  top_db: ~ 

num_works: 32
wav_scp: ./wav.scp
h5:
  output_folder: './fbank/'
  prefix: 'fb-40' # not lfb!!!
  n_wavs: 10240
```

As shown above, the fbank features are configured in the ```fbank``` section and the resulting features are stored into multiple hdf5 files, each of which contains features extracted form  10240 wave files. You can change ```n_wavs``` to -1 to disable grouping. However, doing so will result in a very large file if you have a lot of wave files to process.
