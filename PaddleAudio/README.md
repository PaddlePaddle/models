# PaddleAudio
Unofficial  paddle audio codebase

## Install
```
git clone https://github.com/ranchlai/PaddleAudio.git
cd PaddleAudio
pip install .

```

## Usage
```
import paddleAudio as pa
s,r = pa.load(f)
mel = pa.features.mel_spect(s,r)
```
## to do 


- add sound effects(tempo, mag, etc) , sox supports
- add dataset support
- add models DCASE classication ASD，sound classification
- add demos (audio,video demos) 
- add openL3 support

