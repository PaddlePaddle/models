# Google Landmark Retrieval and Recognition 2019
The Google Landmark Dataset V2 is currently the largest publicly image retrieval and recogntion dataset, including 4M training data, more than 100,000 query images and nearly 1M index data. The large amounts of images in training dataset is the driving force of the generalizability of machine learning models. Here, we release our trained models in Google Landmark 2019 Competition, the detail of our solution can refer to our paper [[link](https://arxiv.org/pdf/1906.03990.pdf)].

## Retrieval Models

We fine-tune four convolutional neural networks to extract our global image descriptors. The four convolutional backbones include ResNet152, ResNet200, SE ResNeXt152 and InceptionV4. We choose arcmargin and npairs as our training loss, We train these models using Google Landmark V2 training set and index set. You can download trained models here. The training code can refer to metric learning [[link](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning)].

|model | public | private
|- | - | -:
|[res152_arcmargin](https://landmark.gz.bcebos.com/res152_arcmargin.tar) | 0.2676 | 0.3020
|[res152_arcmargin_index](https://landmark.gz.bcebos.com/res152_arcmargin_index.tar) | 0.2476 | 0.2707
|[res152_npairs](https://landmark.gz.bcebos.com/res152_npairs.tar) | 0.2597 |  0.2870
|[res200_arcmargin](https://landmark.gz.bcebos.com/res200_arcmargin.tar) | 0.2670 | 0.3042
|[se_x152_arcmargin](https://landmark.gz.bcebos.com/se_x152_arcmargin.tar) | 0.2670 |  0.2914
|[inceptionv4_arcmargin](https://landmark.gz.bcebos.com/inceptionv4_arcmargin.tar) | 0.2685 | 0.2933

In addition, we also train a classification model based on ResNet152 with ~4M Google Landmark V2 training set. ([res152_softmax_v1](https://landmark.gz.bcebos.com/res152_softmax_v1.tar)) 
The taining code can refer to image classification [[link](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification)].

## Recognition Models

There are three models in our recognition solution.

1.[res152_arcmargin](https://landmark.gz.bcebos.com/res152_arcmargin.tar): Retrieval model based on Resnet152 and arcmargin which is the same as in the retrieval task.  

2.[res152_softmax_v2](https://landmark.gz.bcebos.com/res152_softmax_v2.tar): Classification model based on Resnet152 and softmax with ~3M Google Landmark V2 tidied training set. The training code can refer to image classification [[link](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification)].

3.[res50_oid_v4_detector](https://landmark.gz.bcebos.com/res50_oid_v4_detector.tar): Object detector model for the non-landmark images filtering. The mAP of this model is ~0.55 on the OID V4 track (public LB). The training code can refer to RCNN detector [[link](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/rcnn)].

## Environment

Cudnn >= 7, CUDA 9, PaddlePaddle version >= 1.3, python version 2.7

## Inference

### 1.Compile paddle infer so and predict with binary model

There are two different type of models in PaddlePaddle: train model and binary model. Predict with the binary model is more efficient. Thus, at first we compile paddle infer so and convert train model to binary model.

(1) Compile paddle infer so 

Please refer the README.md in pypredict. 

(2) Convert train model to binary model

```
    pushd inference
    sh convert.sh
```

### 2.Extract retrieval feature and calculate cosine distance

In the folder ./inference/test_data, there are four images, 0.jpg and 1.jpg are same landmark images, 2.jpg is another landmark image, 3.jpg is a non-lamdnark image.

We will extract the features of these images, and calculate the cosine distances between 0.jpg and 1.jpg, 2.jpg, 3.jpg.

```
pushd inference
. set_env.sh
python infer_retrieval.py test_retrieval model_name [res152_arcmargin, res152_arcmargin_index, res152_npairs, res200_arcmargin, se_x152_arcmargin, inceptionv4_arcmargin]

example:
    python infer_retrieval.py test_retrieval res152_arcmargin
popd
```

### 3.Predict the classification label of images

```
pushd inference
. set_env.sh
python infer_recognition.py test_cls img_path model_name [res152_softmax_v1, res152_softmax_v2]

example:
    python infer_recognition.py test_cls test_data/0.jpg res152_softmax_v1
popd
```
You will get the inference label and score.

### 4.Detect images

```
    pushd inference
    . set_env.sh
    python infer_recognition.py test_det ./test_data/2e44b31818acc600.jpeg
```

You will get the inference detetor bounding box and classes. The class mapping file: pretrained_models/res50_oid_v4_detector/cls_name_idx_map_openimagev4_500.txt 
