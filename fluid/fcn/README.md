**Fully Convolutional Networks for Semantic Segmentation**
---

**Introduction**
---
FCN[1]（Fully Convolutional Networks) is one of the pioneering work in semantic segmentation. This example demonstrates how to use the FCN model in PaddlePaddle for image segmentation. We first provide a brief introduction to the FCN principle, and then describe how to train and evaluate the model in PASCAL VOC dataset.

**FCN Architecture**
---
FCN is an end-to-end network for semantic segmentation, it takes the input image and with a froward propagation, the output is the predicted result. FCN is based on VGG16[2], but differs as following:
1. Convert the fully connected layers into fully convolutional layers, so as to take input of arbitrary size.
2. The deconvolutional layer is used to upsample the feature map to the input dimensions.
3. The skip-connection architecture is defined to combine deep, coarse, semantic information and shallow, fine, apperance information.

The overall structure of FCN is shown below:
![FCN_ARCH](https://github.com/chengyuz/models/blob/yucheng/fluid/fcn/images/fcn_network.png?raw=true)

FCN learns to combine coarse, high layer information with fine, low layer information. Layers are shown as grids that reveal relative spatial coarseness. Only pooling and prediction layers are shown, intermediate convolutional layers are omitted. FCN-32s upsamples stride 32 predictions back to pixels in a single step. FCN-16s combines predictions from both the final layer and the pool4 layer, at stride 16, so the net predict finer details, while retaining high-level semantic information. FCN-8s adds predictions from pool3, at stride 8, provide further precision.

**Example Overview**
---
This example contains the following files:

Table 1. Directory structure

File                              | Description                                    |
-------------------------         | -------------------------------------   |
train.py                          | Training script                                |  
infer.py                          | Prediction using the trained model      |  
vgg_fcn.py                        | Defining FCN structure                     |  
data_provider.py                  | Data processing scripts, generating train and test data   |  
utils.py                          | Contains common functions                            |  
data/prepare_voc_data.py          | Prepare PASCAL VOC data list for training and test   |

**PASCAL VOC Data set**
---
**Data Preparation**

First download the data set: [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)[3] train dataset and [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)[4] test dataset, and then unzip the data as `data/VOCdevkit/VOC2012` and `data/VOCdevkit/VOC2007`.

Next, run `python prepare_voc_data.py` to generate `voc2012_trainval.txt` and `voc2007_test.txt`.

The data in `voc2012_trainval.txt` will look like:
```
VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg voc_processed/2007_000032.png
VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg voc_processed/2007_000033.png
VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg voc_processed/2007_000039.png
```
The data in `voc2007_test.txt` will look like:
```
VOCdevkit/VOC2007/JPEGImages/000068.jpg
VOCdevkit/VOC2007/JPEGImages/000175.jpg
VOCdevkit/VOC2007/JPEGImages/000243.jpg
```

**To Use Pre-trained Model**

We also provide a pre-trained model of VGG16. To use the model, download the file: [VGG16](https://pan.baidu.com/s/1ekZ5O-lp3lGvAOZ4KSXKDQ) and place it in: `models/vgg16_weights.tar`, and then unzip.

**Training**

Next, run `python train.py --fcn_arch fcn-32s` to train the FCN-32s model, we also provide model structure of FCN-16s and FCN-8s. The relevant function is as following:
```python
weights_dict = resolve_caffe_model(args.pretrain_model)
for k, v in weights_dict.items():
    _tensor = fluid.global_scope().find_var(k).get_tensor()
    _shape = np.array(_tensor).shape
    _tensor.set(v, place)

data_args = data_provider.Settings(
        data_dir=args.data_dir,
        resize_h=args.img_height,
        resize_w=args.img_width,
        mean_value=mean_value)
```
Below is the description about this script:
1. Call `resolve_caffe_model` to get the pre-trained model parameters, and then use the `set` function in fluid to initialize the model.
2. Call `data_provider.Settings` to pass configuration parameters, which can be set by command line.
3. Call `fluid.io.save_inference_model` to save the model per epoch.

Below is the training loss of FCN-32s, FCN-16s and FCN-8s in VOC dataset.
![FCN_LOSS](https://github.com/chengyuz/models/blob/yucheng/fluid/fcn/images/train_loss.png?raw=true)

**Model Assessment**

Run `python infer.py` to evaluate the trained model, the predicted result is save in `demo` directory, which can be set by `--vis_dir` in command line. The relevant function is as following:
```python
model_dir = os.path.join(args.model_dir, '%s-model' % args.fcn_arch)
assert(os.path.exists(model_dir))
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_dir, exe)

predict = exe.run(inference_program, feed={feed_target_names[0]:img_data}, fetch_list=fetch_targets)
res = np.argmax(np.squeeze(predict[0]), axis=0)
res = convert_to_color_label(res)
```
Description:
the `fluid.io.load_inference_model` is called to load the trained model, the `convert_to_color_label` function is used to visualize the predicted as VOC format.

Below is the segmentation result of FCN-32s, FCN-16s and FCN-8s:
![FCN-32s-seg](https://github.com/chengyuz/models/blob/yucheng/fluid/fcn/images/seg_res.png?raw=true)

We provide the trained FCN model:
[FCN-32s](https://pan.baidu.com/s/1j8pltdzgssmxbXFgHWmCNQ)[Password: dk0i]
[FCN-16s](https://pan.baidu.com/s/1idapCRSxWsJKSqqswUGDSw)(Password: q8gu)
[FCN-8s](https://pan.baidu.com/s/1GcO-mcOWo_VF65X3xwPnpA)(Password: du9x)

**References**
---
1. Jonathan Long, Evan Shelhamer, Trevor Darrell. [Fully convolutional networks for semantic segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). IEEE conference on computer vision and pattern recognition, 2015.
2. Simonyan, Karen, and Andrew Zisserman. [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/abs/1409.1556). arXiv preprint arXiv:1409.1556 (2014).
3. [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
4. [The PASCAL Visual Object Classes Challenge 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
