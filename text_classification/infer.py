import sys
import os
import gzip

import paddle.v2 as paddle

import reader
from network_conf import fc_net, convolution_net
from utils import logger, load_dict, load_reverse_dict


def infer(topology, data_dir, model_path, word_dict_path, label_dict_path,
          batch_size):
    def _infer_a_batch(inferer, test_batch, ids_2_word, ids_2_label):
        probs = inferer.infer(input=test_batch, field=["value"])
        assert len(probs) == len(test_batch)
        for word_ids, prob in zip(test_batch, probs):
            word_text = " ".join([ids_2_word[id] for id in word_ids[0]])
            print("%s\t%s\t%s" % (ids_2_label[prob.argmax()],
                                  " ".join(["{:0.4f}".format(p)
                                            for p in prob]), word_text))

    logger.info("begin to predict...")
    use_default_data = (data_dir is None)

    if use_default_data:
        word_dict = paddle.dataset.imdb.word_dict()
        word_reverse_dict = dict((value, key)
                                 for key, value in word_dict.iteritems())
        label_reverse_dict = {0: "positive", 1: "negative"}
        test_reader = paddle.dataset.imdb.test(word_dict)()
    else:
        assert os.path.exists(
            word_dict_path), "the word dictionary file does not exist"
        assert os.path.exists(
            label_dict_path), "the label dictionary file does not exist"

        word_dict = load_dict(word_dict_path)
        word_reverse_dict = load_reverse_dict(word_dict_path)
        label_reverse_dict = load_reverse_dict(label_dict_path)

        test_reader = reader.test_reader(data_dir, word_dict)()

    dict_dim = len(word_dict)
    class_num = len(label_reverse_dict)
    prob_layer = topology(dict_dim, class_num, is_infer=True)

    # initialize PaddlePaddle
    paddle.init(use_gpu=False, trainer_count=1)

    # load the trained models
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(model_path, "r"))
    inferer = paddle.inference.Inference(
        output_layer=prob_layer, parameters=parameters)

    test_batch = []
    for idx, item in enumerate(test_reader):
        test_batch.append([item[0]])
        if len(test_batch) == batch_size:
            _infer_a_batch(inferer, test_batch, word_reverse_dict,
                           label_reverse_dict)
            test_batch = []

    if len(test_batch):
        _infer_a_batch(inferer, test_batch, word_reverse_dict,
                       label_reverse_dict)
        test_batch = []


if __name__ == "__main__":
    model_path = "models/dnn_params_pass_00000.tar.gz"
    assert os.path.exists(model_path), "the trained model does not exist."

    nn_type = "dnn"
    test_dir = None
    word_dict = None
    label_dict = None

    if nn_type == "dnn":
        topology = fc_net
    elif nn_type == "cnn":
        topology = convolution_net

    infer(
        topology=topology,
        data_dir=test_dir,
        word_dict_path=word_dict,
        label_dict_path=label_dict,
        model_path=model_path,
        batch_size=10)
