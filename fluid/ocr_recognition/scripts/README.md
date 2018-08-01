## Introduction
Scripts enclosed in the folder serve as examples of commands that start training
and inference of a model, and are subject to further customisation.

# Running with MKL-DNN
In order to run training or inference using MKL-DNN library, please use
`FLAGS_use_mkldnn=1` environmental variable.

## Prerequisites
In order to run the training and inference, no special requirements are posed.

## Training
To run training on *CPU*, please execute:

```sh
source train.sh CPU
```

To run training on *CPU* with MKL-DNN, please execute:

```sh
source train.sh MKLDNN
```

To run training on *GPU*, please execute:

```sh
source train.sh GPU
```

## Inference
To perform inference on the trained model using *CPU*, please run:
```sh
source infer.sh CPU
```

To perform inference on the trained model using *CPU* with MKL-DNN, please run:
```sh
source infer.sh MKLDNN
```

To perform inference on the trained model using *GPU*, please run:

```sh
source infer.sh GPU
```
