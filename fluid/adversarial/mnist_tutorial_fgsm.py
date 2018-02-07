"""
FGSM demos on mnist using advbox tool.
"""
import matplotlib.pyplot as plt
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

from advbox import Adversary
from advbox.attacks.gradientsign import GradientSignAttack
from advbox.models.paddle import PaddleModel


def cnn_model(img):
    """
    Mnist cnn model
    Args:
        img(Varaible): the input image to be recognized
    Returns:
        Variable: the label prediction
    """
    # conv1 = fluid.nets.conv2d()
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        num_filters=20,
        filter_size=5,
        pool_size=2,
        pool_stride=2,
        act='relu')

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        num_filters=50,
        filter_size=5,
        pool_size=2,
        pool_stride=2,
        act='relu')

    logits = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return logits


def main():
    """
    Advbox demo which demonstrate how to use advbox.
    """
    IMG_NAME = 'img'
    LABEL_NAME = 'label'

    img = fluid.layers.data(name=IMG_NAME, shape=[1, 28, 28], dtype='float32')
    # gradient should flow
    img.stop_gradient = False
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')
    logits = cnn_model(img)
    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    BATCH_SIZE = 1
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(
        feed_list=[IMG_NAME, LABEL_NAME],
        place=place,
        program=fluid.default_main_program())

    fluid.io.load_params(
        exe, "./mnist/", main_program=fluid.default_main_program())

    # advbox demo
    m = PaddleModel(fluid.default_main_program(), IMG_NAME, LABEL_NAME,
                    logits.name, avg_cost.name, (-1, 1))
    att = GradientSignAttack(m)
    for data in train_reader():
        # fgsm attack
        adversary = att(Adversary(data[0][0], data[0][1]))
        if adversary.is_successful():
            plt.imshow(adversary.target, cmap='Greys_r')
            plt.show()
            # np.save('adv_img', adversary.target)
        break


if __name__ == '__main__':
    main()
