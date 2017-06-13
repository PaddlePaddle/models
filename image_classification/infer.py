import gzip
import paddle.v2 as paddle
import reader
import vgg
import resnet
import alexnet
import googlenet
import argparse
import os
from PIL import Image
import numpy as np

WIDTH = 224
HEIGHT = 224
DATA_DIM = 3 * WIDTH * HEIGHT
CLASS_DIM = 102


def main():
    # parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_list',
        help='The path of data list file, which consists of one image path per line'
    )
    parser.add_argument(
        'model',
        help='The model for image classification',
        choices=['alexnet', 'vgg13', 'vgg16', 'vgg19', 'resnet', 'googlenet'])
    parser.add_argument(
        'params_path', help='The file which stores the parameters')
    args = parser.parse_args()

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))

    if args.model == 'alexnet':
        out = alexnet.alexnet(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg13':
        out = vgg.vgg13(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg16':
        out = vgg.vgg16(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg19':
        out = vgg.vgg19(image, class_dim=CLASS_DIM)
    elif args.model == 'resnet':
        out = resnet.resnet_imagenet(image, class_dim=CLASS_DIM)
    elif args.model == 'googlenet':
        out, _, _ = googlenet.googlenet(image, class_dim=CLASS_DIM)

    # load parameters
    with gzip.open(args.params_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    def load_image(file):
        im = Image.open(file)
        im = im.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        im = np.array(im).astype(np.float32)
        # The storage order of the loaded image is W(widht),
        # H(height), C(channel). PaddlePaddle requires
        # the CHW order, so transpose them.
        im = im.transpose((2, 0, 1))  # CHW
        # In the training phase, the channel order of CIFAR
        # image is B(Blue), G(green), R(Red). But PIL open
        # image in RGB mode. It must swap the channel order.
        im = im[(2, 1, 0), :, :]  # BGR
        im = im.flatten()
        im = im / 255.0
        return im

    file_list = [line.strip() for line in open(args.data_list)]
    test_data = [(load_image(image_file), ) for image_file in file_list]
    probs = paddle.infer(
        output_layer=out, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)
    for file_name, result in zip(file_list, lab):
        print "Label of %s is: %d" % (file_name, result[0])


if __name__ == '__main__':
    main()
