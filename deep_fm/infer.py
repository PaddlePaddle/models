import os
import gzip
import argparse
import itertools

import paddle.v2 as paddle

from network_conf import DeepFM
import reader


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DeepFM example")
    parser.add_argument(
        '--model_gz_path',
        type=str,
        required=True,
        help="The path of model parameters gz file")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="The path of the dataset to infer")
    parser.add_argument(
        '--prediction_output_path',
        type=str,
        required=True,
        help="The path to output the prediction")
    parser.add_argument(
        '--factor_size',
        type=int,
        default=10,
        help="The factor size for the factorization machine (default:10)")

    return parser.parse_args()


def infer():
    args = parse_args()

    paddle.init(use_gpu=False, trainer_count=1)

    model = DeepFM(args.factor_size, infer=True)

    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(args.model_gz_path, 'r'))

    inferer = paddle.inference.Inference(
        output_layer=model, parameters=parameters)

    dataset = reader.Dataset()

    infer_reader = paddle.batch(dataset.infer(args.data_path), batch_size=1000)

    with open(args.prediction_output_path, 'w') as out:
        for id, batch in enumerate(infer_reader()):
            res = inferer.infer(input=batch)
            predictions = [x for x in itertools.chain.from_iterable(res)]
            out.write('\n'.join(map(str, predictions)) + '\n')


if __name__ == '__main__':
    infer()
