# PaddleAudio:  The audio library for PaddlePaddle

## Introduction
PaddleAudio is the audio toolkit to speed up your audio research and development loop in PaddlePaddle. It currently provides a collection of audio datasets, feature-extraction functions, audio transforms,state-of-the-art pre-trained models in sound tagging/classification and anomaly sound detection. More models and features are on the roadmap.



## Features
- Spectrogram and related features are compatible with librosa.
- State-of-the-art models in sound tagging on Audioset, sound classification on esc50, and more to come.
- Ready-to-use audio embedding with a line of code, includes sound embedding and more on the roadmap.
- Data loading supports for common open source audio in multiple languages including English, Mandarin and so on.


## Install
```
git clone https://github.com/PaddlePaddle/models
cd models/PaddleAudio
pip install .

```

## Quick start
### Audio loading and feature extraction
``` python
import paddleaudio

audio_file = 'test.flac'
wav, sr = paddleaudio.load(audio_file, sr=16000)
mel_feature = paddleaudio.melspectrogram(wav,
                                       sr=sr,
                                       window_size=320,
                                       hop_length=160,
                                       n_mels=80)
```

### Speech recognition using wav2vec 2.0
``` python
import paddleaudio
from paddleaudio.models.wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

model = Wav2Vec2ForCTC('wav2vec2-base-960h', pretrained=True)
tokenizer = Wav2Vec2Tokenizer()
# Load audio and normalize
wav, _ = paddleaudio.load('your_audio.wav', sr=16000, normal=True, norm_type='gaussian')

with paddle.no_grad():
    x = paddle.to_tensor(wav)
    logits = model(x.unsqueeze(0))
    # Get the token index prediction
    idx = paddle.argmax(logits, -1)
    # Decode prediction to text
    text = tokenizer.decode(idx[0])
    print(text)

```

###  Examples
We provide a set of examples to help you get started in using PaddleAudio quickly.

- [Wav2vec 2.0 for speech recognition](./examples/wav2vec2)
- [PANNs:  acoustic scene and events analysis using pre-trained models](./examples/panns)
- [Environmental Sound classification on ESC-50 dataset](./examples/sound_classification)
- [Training a audio-tagging network on Audioset](./examples/audioset_training)

Please refer to [example directory](./examples) for more details.
