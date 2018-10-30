# Video Classification Based on Temporal Segment Network

Video classification has drawn a significant amount of attentions in the past few years. This page introduces how to perform video classification with PaddlePaddle Fluid, on the public UCF-101 dataset, based on the state-of-the-art Temporal Segment Network (TSN) method.

______________________________________________________________________________

## Table of Contents
<li>Installation</li>
<li>Data preparation</li>
<li>Training</li>
<li>Evaluation</li>
<li>Inference</li>
<li>Performance</li>

### Installation
Running sample code in this directory requires PaddelPaddle Fluid v0.13.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in <a href="http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html" rel="nofollow">installation document</a> and make an update.

### Data preparation

#### download UCF-101 dataset
Users can download the UCF-101 dataset by the provided script in <code>data/download.sh</code>.

#### decode video into frame
To avoid the process of decoding videos in network training, we offline decode them into frames and save it in the <code>pickle</code> format, easily readable for python.

Users can refer to the script <code>data/video_decode.py</code> for video decoding.

#### split data into train and test
We follow the split 1 of UCF-101 dataset. After data splitting, users can get 9537 videos for training and 3783 videos for validation. The reference script is <code>data/split_data.py</code>.

#### save pickle for training
As stated above, we save all data as <code>pickle</code> format for training. All information in each video is saved into one pickle, includes video id, frames binary and label. Please refer to the script <code>data/generate_train_data.py</code>.
After this operation, one can get two directories containing training and testing data in <code>pickle</code> format, and two files <em>train.list</em> and <em>test.list</em>, with each line seperated by SPACE.

### Training
After data preparation, users can start the PaddlePaddle Fluid training by:
```
python train.py \
    --batch_size=128 \
    --total_videos=9537 \
    --class_dim=101 \
    --num_epochs=60 \
    --image_shape=3,224,224 \
    --model_save_dir=output/ \
    --with_mem_opt=True \
    --lr_init=0.01 \
    --num_layers=50 \
    --seg_num=7 \
    --pretrained_model={path_to_pretrained_model}
```

<strong>parameter introduction:</strong>
<li>batch_size: the size of each mini-batch.</li>
<li>total_videos: total number of videos in the training set.</li>
<li>class_dim: the class number of the classification task.</li>
<li>num_epochs: the number of epochs.</li>
<li>image_shape: input size of the network.</li>
<li>model_save_dir: the directory to save trained model.</li>
<li>with_mem_opt: whether to use memory optimization or not.</li>
<li>lr_init: initialized learning rate.</li>
<li>num_layers: the number of layers for ResNet.</li>
<li>seg_num: the number of segments in TSN.</li>
<li>pretrained_model: model path for pretraining.</li>
</br>

<strong>data reader introduction:</strong>
Data reader is defined in <code>reader.py</code>. Note that we use group operation for all frames in one video.


<strong>training:</strong>
The training log is like:
```
[TRAIN] Pass: 0    trainbatch: 0    loss: 4.630959    acc1: 0.0    acc5: 0.0390625    time: 3.09 sec
[TRAIN] Pass: 0    trainbatch: 10    loss: 4.559069    acc1: 0.0546875    acc5: 0.1171875    time: 3.91 sec
[TRAIN] Pass: 0    trainbatch: 20    loss: 4.040092    acc1: 0.09375    acc5: 0.3515625    time: 3.88 sec
[TRAIN] Pass: 0    trainbatch: 30    loss: 3.478214    acc1: 0.3203125    acc5: 0.5546875    time: 3.32 sec
[TRAIN] Pass: 0    trainbatch: 40    loss: 3.005404    acc1: 0.3515625    acc5: 0.6796875    time: 3.33 sec
[TRAIN] Pass: 0    trainbatch: 50    loss: 2.585245    acc1: 0.4609375    acc5: 0.7265625    time: 3.13 sec
[TRAIN] Pass: 0    trainbatch: 60    loss: 2.151489    acc1: 0.4921875    acc5: 0.8203125    time: 3.35 sec
[TRAIN] Pass: 0    trainbatch: 70    loss: 1.981680    acc1: 0.578125    acc5: 0.8359375    time: 3.30 sec
```

### Evaluation
Evaluation is to evaluate the performance of a trained model. One can download pretrained models and set its path to path_to_pretrain_model. Then top1/top5 accuracy can be obtained by running the following command:
```
python eval.py \
    --batch_size=128 \
    --class_dim=101 \
    --image_shape=3,224,224 \
    --with_mem_opt=True \
    --num_layers=50 \
    --seg_num=7 \
    --test_model={path_to_pretrained_model}
```

According to the congfiguration of evaluation, the output log is like:
```
[TEST] Pass: 0    testbatch: 0    loss: 0.011551    acc1: 1.0    acc5: 1.0    time: 0.48 sec
[TEST] Pass: 0    testbatch: 10    loss: 0.710330    acc1: 0.75    acc5: 1.0    time: 0.49 sec
[TEST] Pass: 0    testbatch: 20    loss: 0.000547    acc1: 1.0    acc5: 1.0    time: 0.48 sec
[TEST] Pass: 0    testbatch: 30    loss: 0.036623    acc1: 1.0    acc5: 1.0    time: 0.48 sec
[TEST] Pass: 0    testbatch: 40    loss: 0.138705    acc1: 1.0    acc5: 1.0    time: 0.48 sec
[TEST] Pass: 0    testbatch: 50    loss: 0.056909    acc1: 1.0    acc5: 1.0    time: 0.49 sec
[TEST] Pass: 0    testbatch: 60    loss: 0.742937    acc1: 0.75    acc5: 1.0    time: 0.49 sec
[TEST] Pass: 0    testbatch: 70    loss: 1.720186    acc1: 0.5    acc5: 0.875    time: 0.48 sec
[TEST] Pass: 0    testbatch: 80    loss: 0.199669    acc1: 0.875    acc5: 1.0    time: 0.48 sec
[TEST] Pass: 0    testbatch: 90    loss: 0.195510    acc1: 1.0    acc5: 1.0    time: 0.48 sec
```

### Inference
Inference is used to get prediction score or video features based on trained models.
```
python infer.py \
    --class_dim=101 \
    --image_shape=3,224,224 \
    --with_mem_opt=True \
    --num_layers=50 \
    --seg_num=7 \
    --test_model={path_to_pretrained_model}
```

The output contains predication results, including maximum score (before softmax) and corresponding predicted label.
```
Test sample: PlayingGuitar_g01_c03, score: [21.418629], class [62]
Test sample: SalsaSpin_g05_c06, score: [13.238657], class [76]
Test sample: TrampolineJumping_g04_c01, score: [21.722862], class [93]
Test sample: JavelinThrow_g01_c04, score: [16.27892], class [44]
Test sample: PlayingTabla_g01_c01, score: [15.366951], class [65]
Test sample: ParallelBars_g04_c07, score: [18.42596], class [56]
Test sample: PlayingCello_g05_c05, score: [18.795723], class [58]
Test sample: LongJump_g03_c04, score: [7.100088], class [50]
Test sample: SkyDiving_g06_c03, score: [15.144707], class [82]
Test sample: UnevenBars_g07_c04, score: [22.114838], class [95]
```

### Performance
Configuration | Top-1 acc
------------- | ---------------:
seg=7,  size=224 | 0.859
seg=10, size=224 | 0.863
