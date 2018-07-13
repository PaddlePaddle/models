**Stacked Denoising Autoencoders**
---

**Introduction**
---
The stacked denoising autoencoders(SDA)[1] is stacked by denoising autoencoder. SDA to initialize a deep network is much the same way as stacking RBMs in deep belief networks. Let us specify that input corruption is only used for the initial denoising-training of each individual layer, it may learn useful feature extractors. Once the mapping has thus been learnt, it will hence force be used on uncorrupted inputs. In particular no corruption is applied to produce the representation that will serve as clean input for training the next layer.

**Autoencoder Architecture**
---
The architecture of SDA is based on autoencoder, there are two parts in autoencoder: encoder and decoder. The encoder transforms the input d-dimensional vector $\textbf{x}$ into hidden representation:
$$\textbf{y}=s(\textbf{Wx}+\textbf{b})$$
where $\textbf{W}$ is the weight matrix and the $\textbf{b}$ is the bias. The resulting hidden representation is then mapped back to a reconstructed d-dimensional vector:
$$\textbf{z}=s(\textbf{W}'\textbf{y}+\textbf{b}')$$
The autoencoder is trained with cross entropy loss function：
$$L(\textbf{x},\textbf{z})=-\sum_{k=1}^{d}[\textbf{x}_klog\textbf{z}_k+(1-\textbf{x}_k)log(1-\textbf{z}_k)]$$

**Example Overview**
---

This example contains the following files:

Table 1. Directory structure

 File                              | Description                              |
 -------------------------         | -------------------------------------   |
 autoencoder.py    | Autoencoder definition script                      |  
 train_da.py       | Autoencoder training script      |  
 stacked_autoencoder.py| SDA definition script            |  
 train_sda.py     | SDA training script        |  
 utils.py         |  Contains common functions   |  
 infer.py          | Prediction using the trained model           |  
 train_sda.sh      | SDA training shell script  |


**Experiment**
---
**Denoising Autoencoders**

The goal of this part is to better understand the qualitative effect of noise level. So we trained several denoising autoencoders, all start from the same initial random point in weight space, but with different noise level. For this experiment, we use denoising autoencoders with tied weights, cross-entropy reconstruction error, and zero-maksing noise, the experiment is based on MNIST dataset.

Run `python train_da.py` to train denoising autoencoders, and visualize the learned filters. The definition of autoencoder (autoencoder.py)is like:
```python
def denoise_autoencoder(input, args):
    n_hidden = args.n_hidden
    n_visible = args.img_height * args.img_width
    W = fluid.layers.create_parameter(shape=[n_visible, n_hidden], dtype='float32', attr=fluid.ParamAttr(name='W', initializer=fluid.initializer.Normal()), is_bias=False)
    bvis = fluid.layers.zeros(shape=[n_visible], dtype='float32')
    bhid = fluid.layers.zeros(shape=[n_hidden], dtype='float32')
    hidden_value = fluid.layers.sigmoid(fluid.layers.matmul(input, W) + bhid)
    out = fluid.layers.sigmoid(fluid.layers.matmul(hidden_value, W, transpose_y=True) + bvis)
    return out
```
Below is the result of learned filter. As can be seen, with no noise, many filters remain similary uninteresting, as we increase the noise level, denoising training forces the filters to differentiate more, and capture more distinctive features.
![The filters learned by denoising autoencoders](https://github.com/chengyuz/models/blob/yucheng_sda/fluid/sda/images/da_res.png)

**Stacked Denoising Autoencoders**

In this section, we evaluate denoising autoencoders as a pretraining strategy for building deep networks, using stacking procedure. We will mainly compare the classification performance of networks pretrained by stacking denoising autoencoders(SDAE), versus stacking regular autoencoders(SAE). The experiment is based on MNIST dataset, and zero-masking corruption noise is added.

The definition of stacking autoencoders is like:
```python
def build_model(self, layer_input):
    for i in range(len(self.num_layers)):
        sigmoid_layer = fluid.layers.fc(
            input=layer_input,
            size=self.num_layers[i],
            act='sigmoid',
            param_attr=fluid.ParamAttr(
                name='s%d_w' % i, initializer=fluid.initializer.Normal()),
            bias_attr=fluid.ParamAttr(
                name='s%d_b' % i, initializer=fluid.initializer.Constant()))
        layer_input = sigmoid_layer

    out = fluid.layers.fc(
        input=sigmoid_layer,
        size=self.class_num,
        act='softmax',
        param_attr=fluid.ParamAttr(
            name='out_w', initializer=fluid.initializer.Normal()),
        bias_attr=fluid.ParamAttr(
            name='out_b', initializer=fluid.initializer.Constant()))
    return out
```
Specifically:
1. Run `train_sda.sh` to train SDA, the trained model is saved in `models/SDAE`
2. Run `python train_sda.py --mode sda --pretrain_strategy SAE` to train SAE, the trained model is saved in `models/SAE`
3. Run `python infer.py --mode SDAE` to evaluate SDA model
4. Run `python infer.py --mode SAE` to evaluate SAE model

The classification result is shown as following:

 Model                  | Top1 Accuracy                   |
 -------------------------         | -------------------------------------   |
 SDA    |          0.967             |
SAE|  0.961 |

We can see the top1 accuracy of SDA is better than SAE, so denoising pretraining being better than no pretraining.

---

**References**
1. Jonathan Long, Evan Shelhamer, Trevor Darrell. [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](http://www.jmlr.org/papers/v11/vincent10a.html), JMLR2010.
