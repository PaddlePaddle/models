# Speaker verification using and ResnetSE ECAPA-TDNN

## Introduction
In this example, we demonstrate how to use PaddleAudio to train two types of networks for speaker verification.
The networks supported here are
- Resnet34 with Squeeze-and-excite block \[1\] to adaptively re-weight the feature maps.
- ECAPA-TDNN  \[2\]

## Requirements
Install the requirements via
```
# install paddleaudio
git clone https://github.com/PaddlePaddle/models.git
cd models/PaddleAudio
pip install -e .
```
Then install additional requirements by
```
cd examples/speaker
pip install -r requirements.txt
```

## Training
### Training datasets
Following from this example and this example, we use the dev split [VoxCeleb 1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) which consists aof `1,211` speakers and the dev split of [VoxCeleb 2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) consisting of `5,994` speakers for training. Thus there are `7,502` speakers totally in our training set.

Please download the two datasets from the [official website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb) and unzip all audio into a folder, e.g., `./data/voxceleb/`. Make sure there are `7502` subfolders with prefix  `id1****` under the folder. You don't need to further process the data because all data processing such as adding noise / reverberation / speed perturbation  will be done on-the-fly. However, to speed up audio decoding, you can manually convert the m4a file in VoxCeleb 2 to wav file format, at the expanse of using more storage.

Finally, create a txt file that contains the list of audios for training by
```
cd ./data/voxceleb/
find `pwd`/ --type f > vox_files.txt
```
### Augmentation datasets
The following datasets are required for dataset augmentation
- [Room Impulse Response and Noise Database](https://openslr.org/28/)
- [MUSAN](https://openslr.org/17/)

For the RIR dataset, you must list all audio files under the folder `RIRS_NOISES/simulated_rirs/` into a text file, e.g., data/rir.list and config it as rir_path in the `config.yaml` file.

Likewise, you have to config the the following fields in the config file for noise augmentation
``` yaml
muse_speech: <musan_split/speech.list> #replace with your actual path
muse_speech_srn_high: 15.0
muse_speech_srn_low: 12.0
muse_music: <musan_split/music.list> #replace with your actual path
muse_music_srn_high: 15.0
muse_music_srn_low: 5.0
muse_noise: <musan_split/noise.list> #replace with your actual path
muse_noise_srn_high: 15
muse_noise_srn_low: 5.0
```

To train your model from scratch, first create a folder(workspace) by
``` bash
cd egs
mkdir <your_example>
cd <your_example>
cp ../resnet/config.yaml . #Copy an example config to your workspace
```
Then change the config file accordingly to make sure all audio files can be correctly located(including the files used for data augmentation). Also you can change the training and model hyper-parameters to suit your need.

Finally start your training by

``` bash
python ../../train.py -c config.yaml  -d gpu:0
```

## Testing
## <a name="test_dataset"></a>Testing datasets
The testing split of VoxCeleb 1 is used for measuring the performance of speaker verification duration training and after the training completes.  You will need to download the data and unzip into a folder, e.g, `./data/voxceleb/test/`.

Then download the text files which list utterance  pairs to compare and the true labels indicating whether the utterances come from the same speaker. There are multiple trials and we will use [veri_test2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt).

To start testing, first download the checkpoints for resnet or ecapa-tdnn,

| checkpoint |size| eer |
| --------------- | --------------- | --------------- |
| [ResnetSE34 + SAP + CMSoftmax](https://bj.bcebos.com/paddleaudio/models/speaker/resnetse34_epoch92_eer0.00931.pdparams) |26MB | 0.93%|
| [ecapa-tdnn + AAMSoftmax ](https://bj.bcebos.com/paddleaudio/models/speaker/tdnn_amsoftmax_epoch51_eer0.011.pdparams)| 80MB |1.10%|

Then prepare the test dataset as described in [Testing datasets](#test_dataset), and set the following path in the config file,
``` yaml
mean_std_file: ../../data/stat.pd
test_list: ../../data/veri_test2.txt
test_folder: ../../data/voxceleb1/
```

To compute the eer using resnet, run:

``` bash
cd egs/resnet/
python ../../test.py -w <checkpoint path> -c config.yaml  -d gpu:0
```
which will result in eer 0.00931.

for ecapa-tdnn, run:
``` bash
cd egs/ecapa-tdnn/
python ../../test.py -w <checkpoint path> -c config.yaml  -d gpu:0
```
which gives you eer 0.0105.

## Results
We compare our results  with [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

### Pretrained model of voxceleb_trainer
The test list is veri_test2.txt, which can be download from here [VoxCeleb1 (cleaned)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt)

| model |config|checkpoint |eval frames| eer |
| --------------- | --------------- | --------------- |--------------- |--------------- |
| ResnetSE34 + ASP + softmaxproto| - | [baseline_v2_ap](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model)|400|1.06%|
| ResnetSE34 + ASP + softmaxproto| - | [baseline_v2_ap](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model)|all|1.18%|

### This example
| model |config|checkpoint |eval frames| eer |
| --------------- | --------------- | --------------- |--------------- |--------------- |
| ResnetSE34 + SAP + CMSoftmax| [config.yaml](./egs/resent/config.yaml) |[checkpoint](https://bj.bcebos.com/paddleaudio/models/speaker/resnetse34_epoch92_eer0.00931.pdparams) | all|0.93%|
| ECAPA-TDNN + AAMSoftmax | [config.yaml](./egs/ecapa-tdnn/config.yaml) | [checkpoint](https://bj.bcebos.com/paddleaudio/models/speaker/tdnn_amsoftmax_epoch51_eer0.011.pdparams) | all|1.10%|

## References
[1] Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141

[2] Desplanques B, Thienpondt J, Demuynck K. Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification[J]. arXiv preprint arXiv:2005.07143, 2020.
