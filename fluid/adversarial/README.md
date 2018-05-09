The minimum PaddlePaddle version needed for the code sample in this directory is the lastest develop branch. If you are on a version of PaddlePaddle earlier than this, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# Advbox

Advbox is a toolbox to generate adversarial examples that fool neural networks and Advbox can benchmark the robustness of machine learning models.

The Advbox is based on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) Fluid and is under continual development, always welcoming contributions of the latest method of adversarial attacks and defenses.


## Overview
[Szegedy et al.](https://arxiv.org/abs/1312.6199) discovered an intriguing properties of deep neural networks in the context of image classification for the first time. They showed that despite the state-of-the-art deep networks are surprisingly susceptible to adversarial attacks in the form of small perturbations to images that remain (almost) imperceptible to human vision system. These perturbations are found by optimizing the input to maximize the prediction error and the images modified by these perturbations are called as `adversarial examples`. The profound implications of these results triggered a wide interest of researchers in adversarial attacks and their defenses for deep learning in general.

Advbox is similar to [Foolbox](https://github.com/bethgelab/foolbox) and [CleverHans](https://github.com/tensorflow/cleverhans). CleverHans only supports TensorFlow framework while foolbox interfaces with many popular machine learning frameworks such as PyTorch, Keras, TensorFlow, Theano, Lasagne and MXNet. However, these two great libraries don't support PaddlePaddle, an easy-to-use, efficient, flexible and scalable deep learning platform which is originally developed by Baidu scientists and engineers for the purpose of applying deep learning to many products at Baidu.

## Usage
Advbox provides many stable reference implementations of modern methods to generate adversarial examples such as FGSM, DeepFool, JSMA. When you want to benchmark the robustness of your neural networks , you can use the advbox to generate some adversarial examples and benchmark the networks. Some tips of using Advbox:

1. Train a model and save the parameters.
2. Load the parameters which has been trained，then reconstruct the model.
3. Use advbox to generate the adversarial samples.


#### Dependencies
* PaddlePaddle: [the lastest develop branch](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html)
* Python 2.x

#### Structure

Network models, attack method's implements and the criterion that defines adversarial examples are three essential elements to generate adversarial examples. Misclassification is adopted as the adversarial criterion for briefness in Advbox.

The structure of Advbox module are as follows:

    .
    ├── advbox
    |   ├── __init__.py
    |   ├── attack
    |        ├── __init__.py
    |        ├── base.py
    |        ├── deepfool.py
    |        ├── gradient_method.py
    |        ├── lbfgs.py
    |        └── saliency.py
    |   ├── models
    |        ├── __init__.py
    |        ├── base.py
    |        └── paddle.py
    |   └── adversary.py
    ├── tutorials
    |   ├── __init__.py
    |   ├── mnist_model.py
    |   ├── mnist_tutorial_lbfgs.py
    |   ├── mnist_tutorial_fgsm.py
    |   ├── mnist_tutorial_bim.py
    |   ├── mnist_tutorial_ilcm.py
    |   ├── mnist_tutorial_mifgsm.py
    |   ├── mnist_tutorial_jsma.py
    |   └── mnist_tutorial_deepfool.py
    └── README.md

**advbox.attack**

Advbox implements several popular adversarial attacks which search adversarial examples. Each attack method uses a distance measure(L1, L2, etc.) to quantify the size of adversarial perturbations. Advbox is easy to craft adversarial example as some attack methods could perform internal hyperparameter tuning to find the minimum perturbation.

**advbox.model**

Advbox implements interfaces to PaddlePaddle. Additionally, other deep learning framworks such as TensorFlow can also be defined and employed. The module is use to compute predictions and gradients for given inputs in a specific framework.

**advbox.adversary**

Adversary contains the original object, the target and the adversarial examples. It provides the misclassification as the criterion to accept a adversarial example.

## Tutorials
The `./tutorials/` folder provides some tutorials to generate adversarial examples on the MNIST dataset. You can slightly modify the code to apply to other dataset. These attack methods are supported in Advbox:

* [L-BFGS](https://arxiv.org/abs/1312.6199)
* [FGSM](https://arxiv.org/abs/1412.6572)
* [BIM](https://arxiv.org/abs/1607.02533)
* [ILCM](https://arxiv.org/abs/1607.02533)
* [MI-FGSM](https://arxiv.org/pdf/1710.06081.pdf)
* [JSMA](https://arxiv.org/pdf/1511.07528)
* [DeepFool](https://arxiv.org/abs/1511.04599)

## Testing
Benchmarks on a vanilla CNN model.

> MNIST

|  adversarial attacks  |  fooling rate (non-targeted)  | fooling rate (targeted) | max_epsilon | iterations | Strength |
|:-----:| :----: | :---: | :----: | :----: | :----: |
|L-BFGS| --- | 89.2% | --- | One shot | *** |
|FGSM| 57.8% | 26.55% | 0.3 | One shot| *** |
|BIM| 97.4% | --- | 0.1 | 100 | **** |
|ILCM| ---  | 100.0% | 0.1 | 100 | **** |
|MI-FGSM| 94.4% | 100.0% | 0.1 | 100 | **** |
|JSMA| 96.8% | 90.4%| 0.1 | 2000 | *** |
|DeepFool| 97.7% | 51.3% | --- | 100 | **** |

* The strength (higher for more asterisks) is based on the impression from the reviewed literature.

---
## References
* [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199), C. Szegedy et al., arxiv 2014
* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), I. Goodfellow et al., ICLR 2015
* [Adversarial Examples In The Physical World](https://arxiv.org/pdf/1607.02533v3.pdf), A. Kurakin et al., ICLR workshop 2017
* [Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081), Yinpeng Dong et al., arxiv 2018
* [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528), N. Papernot et al., ESSP 2016
* [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/abs/1511.04599), S. Moosavi-Dezfooli et al., CVPR 2016
* [Foolbox: A Python toolbox to benchmark the robustness of machine learning models](https://arxiv.org/abs/1707.04131), Jonas Rauber et al., arxiv 2018
* [CleverHans: An adversarial example library for constructing attacks, building defenses, and benchmarking both](https://github.com/tensorflow/cleverhans#setting-up-cleverhans)
* [Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](https://arxiv.org/abs/1801.00553), Naveed Akhtar, Ajmal Mian, arxiv 2018
