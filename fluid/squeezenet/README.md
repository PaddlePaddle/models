SqueezeNet
===========
Introduction
-----------
Much of the recent research on deep convolutional neural networks (CNNs) has focused on increasing accuracy on computer vision datasets. For a given accuracy level, there typically exist multiple CNN architectures that achieve that accuracy level. Given equivalent accuracy, a CNN architecture with fewer parameters has several advantages:：
* More efficient distributed training.  
* Less overhead when exporting new models to clients.
* Feasible FPGA and embedded deployment.

With this in mind, this paper discovers such an architecture, called SqueezeNet, which reaches equivalent accuracy compared to AlexNet with 1/50 fewer parameters.  

Architecture
-----------
### Architecture Design Strategies
1. Replace `3x3` filters with `1x1` filter：1/9 fewer parameters than before
2. Decrease the number of input channels to `3x3` filters：using squeeze layers
3. Downsample late in the network so that convolution layers have large activation maps: large activation maps (due to delayed downsampling) can lead to higher classification accuracy

Strategies 1 and 2 are about judiciously decreasing the quantity of parameters in a CNN while attempting to preserve accuracy. Strategy 3 is about maximizing accuracy on a limited budget of parameters.
### The Fire Module
![](https://github.com/Panxj/SqueezeNet/raw/master/images/fire_module.jpg)
  * squeeze layer: decrease `3x3` filters and channels
  * expand layer: expand channels through a mix of `1x1` and `3x3` filters with three hyperparameters: <a href="https://www.codecogs.com/eqnedit.php?latex=s_{1X1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_{1X1}" title="s_{1X1}" /></a>(number of filters in the squeeze layer), <a href="https://www.codecogs.com/eqnedit.php?latex=e_{1X1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_{1X1}" title="e_{1X1}" /></a>(number of  `1x1` filters in the expand layer), <a href="https://www.codecogs.com/eqnedit.php?latex=e_{3X3}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_{3X3}" title="e_{3X3}" /></a>(number of `3x3` filters in the expand layer)
  * when using Fire modules, set <a href="https://www.codecogs.com/eqnedit.php?latex=s_{1X1}&space;<&space;e_{1X1}&space;&plus;&space;e_{3X3}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_{1X1}&space;<&space;e_{1X1}&space;&plus;&space;e_{3X3}" title="s_{1X1} < e_{1X1} + e_{3X3}" /></a> to limit the number of input channels to the `3x3` filters.

### The SqueezeNet Architecture
SqueezeNet begins with a standalone convolution layer (conv1), followed by 8 Fire modules (fire2-9), ending with a final conv layer (conv10). Gradually increase the number of filters per fire module from the beginning to the end of the network. SqueezeNet performs max-pooling with a stride of 2 after layers conv1, fire4, fire8, and conv10; these relatively late placements of pooling are per Strategy3. As shown followed, the left one is the Macroarchitectural view of SqueezeNet architecture. The middle and right are SqueezeNet with simple bypass and complex bypass correspondingly.

![](https://github.com/Panxj/SqueezeNet/raw/master/images/architecture.jpg)

Overview
-----------
Tabel 1. Directory structure

|file | description|
|:--- |:---|
train.py | Train script
infer.py | Prediction using the trained model
reader.py| Data reader
squeezenet.py| Model definition
data/val.txt|Validation data list
data/train.txt| Train data list
models/SqueezeNet_v1.1| Parameters converted from caffe model
images/\* | Images used in description and some results
output/\* | Parameters saved in training process

Data Preparation
-----------
First, download the ImageNet dataset. We using ILSVRC 2012(ImageNet Large Scale Visual Recognition Challenge) dataset in which,

* trainning set: 1,281,167 imags + labels
* validation set: 50,000 images + labels
* test set: 100,000 images
```
cd data/
mkdir -p ILSVRC2012/
cd ILSVRC2012/
# get training set
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
# get validation set
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
# prepare directory
tar xf ILSVRC2012_img_train.tar
tar xf ILSVRC2012_img_val.tar

```

All the images are resized to `256 X 256` and then croped into ` 227 X 227 ` randomly when training while are cropped from the center when testing and infering. Finally, subtracting the mean value, here we use  `[104,117,123]` , which is slightly different from the official offer. The relevant function in `reader.py` is as following:
```python
def process_image(sample, mode, color_jitter, rotate):
    img_path = sample[0]
    img = paddle2.image.load_image(img_path)
    img = cv2.resize(img, (DATA_DIM, DATA_DIM), interpolation=cv2.INTER_CUBIC)
    if mode == 'train':
        if rotate: img = rotate_image(img)
        img = paddle2.image.random_crop(img, DATA_DIM)
    else:
        img = paddle2.image.center_crop(img, DATA_DIM)
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if random.randint(0, 1) == 1:
            img = paddle2.image.left_right_flip(img)
    img = paddle2.image.to_chw(img)
    img = img.astype('float32')
    img -= img_mean

    if mode == 'train' or mode == 'test':
        return img, sample[1]
    elif mode == 'infer':
        return [img]
```

The training and validation label files looks like as follwed,

**train_list.txt**: training list of imagenet 2012 classification task, with each line seperated by SPACE.
```
n01440764/n01440764_10026.JPEG 0
n01440764/n01440764_10027.JPEG 0
n01440764/n01440764_10029.JPEG 0
n01440764/n01440764_10040.JPEG 0
n01440764/n01440764_10042.JPEG 0
n01440764/n01440764_10043.JPEG 0
...
```
**val_list.txt**: validation list of imagenet 2012 classification task, with each line seperated by SPACE.
```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
ILSVRC2012_val_00000005.JPEG 516
ILSVRC2012_val_00000006.JPEG 57
...
```
**synset_words.txt**: the semantic label of each class.
```
n01491361 tiger shark, Galeocerdo cuvieri
n01494475 hammerhead, hammerhead shark
n01496331 electric ray, crampfish, numbfish, torpedo
```

Training
-----------
#### 1. Determine the architecture
The author [github](https://github.com/DeepScale/SqueezeNet) provide two versions of SqueezeNet model。 SqueezeNet_v1.0 is the implement one of architecture described in paper while SqueezeNet_v1.1 changes somewhere to reduce the computation by 2.4x with the accuracy preserved. Changes in SqueezeNet_v1.1 is shown as follwed：

Tabel 2. changes in SqueezeNet_v1.1

 | | SqueezeNet_v1.0 | SqueezeNet_v1.1|
 |:---|:---|:---|
 |conv1| 96 filters of resolution 7x7|64 filters of resolution 3x3|
 |pooling layers| pool_{1,4,8} | pool_{1,3,5}|
 |computation| 1.72GFLOPS/image| 0.72 GFOLPS/image:2.4x less computation|
 |ImageNet accuracy| >=80.3% top-5| >=80.3% top-5|

Here, we implement the latter one, SqueezeNet_v1.1.

<!--#### 2. caffe2paddle
To train simply and quickly, we first transfor the [caffe](http://caffe.berkeleyvision.org/)-style parameters into ones can be used in [PaddlePaddle](http://www.paddlepaddle.org/) as the initial model and then we train the model from scratch.
We perfome the parameter conversion according to the method described [here](https://github.com/PaddlePaddle/models/tree/develop/image_classification/caffe2paddle). Our converted parameters are placed under directory `models/squeezenet_weights`.
In `train.py`, we set the parameters converted from caffe into the paddle model as the initial value as followed:
```python
#Load initial caffe-style params
if args.model is not None:
    for layer_name in parameters.keys():
        layer_param_path = os.path.join(args.model,layer_name)
        if os.path.exists(layer_param_path):
            h,w = parameters.get_shape(layer_name)
            parameters.set(layer_name,load_parameter(layer_param_path,h,w))
```
-->
#### 2. train
`python train.py | tee ouput/logs/log.log` to perform model trainning process and record the log.


```python
train_parallel_do(args,
                      learning_rate,
                      batch_size,
                      num_passes,
                      init_model=None,
                      pretrained_model=None,
                      model_save_dir='models',
                      parallel=True,
                      use_nccl=True,
                      lr_strategy=None)
```

1. Call paddle.init with 2 GPUs.
2. `reader()`
3. During the training process, it will print some log information.

Testing
-----------
Run `python eavl.py` to preform model evaluation process using the trained model.
```python
add_arg('batch_size', int, 32, "Minibatch size.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('test_list', str, '', "The testing data lists.")
add_arg('model_dir', str, './models/final', "The model path.")

# Evaluation code
eval(args):
```

Infering
-----------
Run `python infer.py` to perform the image classification using the trained model.
```python
add_arg('batch_size', int, 1, "Minibatch size.")
add_arg('use_gpu', bool,  True, "Whether to use GPU or not.")
add_arg('test_list', str, '', "The testing data lists.")
add_arg('synset_word_list', str, 'data/ILSVRC2012/synset_words.txt', "The label name of data")
add_arg('model_dir', str, 'models/final', "The model path.")
# infer code
infer(args)
```


References
-----------
[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
