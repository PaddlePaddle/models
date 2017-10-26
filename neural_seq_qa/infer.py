import os
import sys
import argparse

import paddle.v2 as paddle

import reader
import utils
import network
import config

from utils import logger


class Infer(object):
    def __init__(self, conf):
        self.conf = conf

        self.settings = reader.Settings(
            vocab=conf.vocab, is_training=False, label_schema=conf.label_schema)

        # init paddle
        # TODO(lipeng17) v2 API does not support parallel_nn yet. Therefore, we
        # can only use CPU currently
        paddle.init(use_gpu=conf.use_gpu, trainer_count=conf.trainer_count)

        # define network
        self.tags_layer = network.inference_net(conf)

    def infer(self, model_path, data_path, output):
        test_reader = paddle.batch(
            paddle.reader.buffered(
                reader.create_reader(data_path, self.settings),
                size=self.conf.batch_size * 1000),
            batch_size=self.conf.batch_size)

        # load the trained models
        parameters = paddle.parameters.Parameters.from_tar(
            utils.open_file(model_path, "r"))
        inferer = paddle.inference.Inference(
            output_layer=self.tags_layer, parameters=parameters)

        def count_evi_ids(test_batch):
            num = 0
            for sample in test_batch:
                num += len(sample[reader.E_IDS])
            return num

        for test_batch in test_reader():
            tags = inferer.infer(
                input=test_batch, field=["id"], feeding=network.feeding)
            evi_ids_num = count_evi_ids(test_batch)
            assert len(tags) == evi_ids_num
            print >> output, ";\n".join(str(tag) for tag in tags) + ";"


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("data_path")
    parser.add_argument("output", help="'-' for stdout")
    return parser.parse_args()


def main(args):
    conf = config.InferConfig()
    conf.vocab = utils.load_dict(conf.word_dict_path)
    logger.info("length of word dictionary is : %d." % len(conf.vocab))

    if args.output == "-":
        output = sys.stdout
    else:
        output = utils.open_file(args.output, "w")

    infer = Infer(conf)
    infer.infer(args.model_path, args.data_path, output)

    output.close()


if __name__ == "__main__":
    main(parse_cmd())
