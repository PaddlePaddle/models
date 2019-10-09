The minimum PaddlePaddle version needed for the code sample in this directory is the lastest develop branch. If you are on a version of PaddlePaddle earlier than this, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

## Deep Automatic Speech Recognition

### Introduction
TBD

### Installation

#### Kaldi
The decoder depends on [kaldi](https://github.com/kaldi-asr/kaldi), install it by flowing its instructions. Then

```shell
export KALDI_ROOT=<absolute path to kaldi>
```

#### Decoder

```shell
git clone https://github.com/PaddlePaddle/models.git
cd models/fluid/DeepASR/decoder
sh setup.sh
```

### Data reprocessing
TBD

### Training
TBD


### Inference & Decoding
TBD

### Question and Contribution
TBD
