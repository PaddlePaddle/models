The minimum PaddlePaddle version needed for the code sample in this directory is the lastest develop branch. If you are on a version of PaddlePaddle earlier than this, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# Advbox

Advbox is a Python toolbox to create adversarial examples that fool neural networks. It requires Python and paddle.

## How to use

1. train a model and save it's parameters. (like fluid_mnist.py)
2. load the parameters which is trained in step1, then reconstruct the model.(like mnist_tutorial_fgsm.py)
3. use advbox to generate the adversarial sample.
