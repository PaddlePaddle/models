import os
import gzip
import argparse
import numpy as np
from PIL import Image

import paddle.v2 as paddle
import drn


DATA_DIM = 3 * 224 * 224
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
        choices=[
            'drn'
        ])
    parser.add_argument(
        'params_path', help='The file which stores the parameters')
    args = parser.parse_args()

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))

    if args.model == 'drn':
        out = drn.drn16(image, class_dim=CLASS_DIM)

    # load parameters
    with gzip.open(args.params_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    file_list = [line.strip() for line in open(args.data_list)]
    test_data = [(paddle.image.load_and_transform(image_file, 256, 224, False)
                  .flatten().astype('float32'), ) for image_file in file_list]
    probs = paddle.infer(
        output_layer=out, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)
    for file_name, result in zip(file_list, lab):
        print "Label of %s is: %d" % (file_name, result[0])


if __name__ == '__main__':
    main()