The minimum PaddlePaddle version needed for the code sample in this directory is v0.11.0. If you are on a version of PaddlePaddle earlier than v0.11.0, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# Deep Factorization Machine for Click-Through Rate prediction

## Introduction
This model implements the DeepFM proposed in the following paper:

```text
@inproceedings{guo2017deepfm,
  title={DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  booktitle={the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={1725--1731},
  year={2017}
}
```

The DeepFm combines factorization machine and deep neural networks to model
both low order and high order feature interactions. For details of the
factorization machines, please refer to the paper [factorization
machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

## Dataset
This example uses Criteo dataset which was used for the [Display Advertising
Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)
hosted by Kaggle.

Each row is the features for an ad display and the first column is a label
indicating whether this ad has been clicked or not. There are 39 features in
total. 13 features take integer values and the other 26 features are
categorical features. For the test dataset, the labels are omitted.

Download dataset:
```bash
cd data && ./download.sh && cd ..
```

## Model
The DeepFM model is composed of the factorization machine layer (FM) and deep
neural networks (DNN). All the input features are feeded to both FM and DNN.
The output from FM and DNN are combined to form the final output. The embedding
layer for sparse features in the DNN shares the parameters with the latent
vectors (factors) of the FM layer.

The factorization machine layer in PaddlePaddle computes the second order
interactions. The following code example combines the factorization machine
layer and fully connected layer to form the full version of factorization
machine:

```python
def fm_layer(input, factor_size):
    first_order = paddle.layer.fc(input=input, size=1, act=paddle.activation.Linear())
    second_order = paddle.layer.factorization_machine(input=input, factor_size=factor_size)
    fm = paddle.layer.addto(input=[first_order, second_order],
                            act=paddle.activation.Linear(),
                            bias_attr=False)
    return fm
```

## Data preparation
To preprocess the raw dataset, the integer features are clipped then min-max
normalized to [0, 1] and the categorical features are one-hot encoded. The raw
training dataset are splited such that 90% are used for training and the other
10% are used for validation during training.

```bash
python preprocess.py --datadir ./data/raw --outdir ./data
```

## Train
The command line options for training can be listed by `python train.py -h`.

To train the model:
```bash
python train.py \
        --train_data_path data/train.txt \
        --test_data_path data/valid.txt \
        2>&1 | tee train.log
```

After training pass 9 batch 40000, the testing AUC is `0.807178` and the testing
cost is `0.445196`.

## Infer
The command line options for infering can be listed by `python infer.py -h`.

To make inference for the test dataset:
```bash
python infer.py \
        --model_gz_path models/model-pass-9-batch-10000.tar.gz \
        --data_path data/test.txt \
        --prediction_output_path ./predict.txt
```
