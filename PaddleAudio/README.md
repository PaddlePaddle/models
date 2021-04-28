# PaddleAudio:  a light-weight audio library for PaddlePaddle

## Introduction
PaddleAudio is a lightweight audio toolkit to speed up your audio research and development loop in PaddlePaddle. It currently provides a collection of audio dataests, feature functions and transforms, preprocessing scripts, and state-of-the-art pretrained models in sound tagging(multi-label)/classifcation, anmolay sound detection. More models and feaetures on audio processing are on the roadmap.



## Features
- Fast and smart audio loading, designed specially for deep networks
- Spectrogram features are compactable with kaildi and librosa
- Use hdf5 as dataest backend for large scale audio data storage and fast retrival, hence enabling faster training
- State-of-the-art(sota) or on-par-with sota audio models in sound tagging on audioset/sound classications on esc50, and more to come
- Ready to use audio embedding with a line of code, includs sound embedding and more. 
- Models and training processing are configurable in yaml file
- Ready to use opensource audio datasets, including English / Mandrin / Cantonese and more


## Install
```
git clone https://github.com/PaddlePaddle/models
cd models/PaddleAudio
pip install .

```

## Quick start
#### Load audio and extrat spectrogram
```
import paddleAudio as pa
s,r = pa.load(f)
mel = pa.features.mel_spect(s,r)
```

#### Sound tagging example 

```


```


#### Dataset exmaple

```


```


