import gzip

import reader
import paddle.v2 as paddle
from network_conf import ner_net
from utils import load_dict, load_reverse_dict


def infer(model_path, batch_size, test_data_file, vocab_file, target_file):
    def _infer_a_batch(inferer, test_data, id_2_word, id_2_label):
        probs = inferer.infer(input=test_data, field=["id"])
        assert len(probs) == sum(len(x[0]) for x in test_data)

        for idx, test_sample in enumerate(test_data):
            start_id = 0
            for w, tag in zip(test_sample[0],
                              probs[start_id:start_id + len(test_sample[0])]):
                print("%s\t%s" % (id_2_word[w], id_2_label[tag]))
            print("\n")
            start_id += len(test_sample[0])

    word_dict = load_dict(vocab_file)
    word_dict_len = len(word_dict)
    word_reverse_dict = load_reverse_dict(vocab_file)

    label_dict = load_dict(target_file)
    label_reverse_dict = load_reverse_dict(target_file)
    label_dict_len = len(label_dict)

    # initialize PaddlePaddle
    paddle.init(use_gpu=False, trainer_count=1)
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(model_path, "r"))

    predict = ner_net(
        word_dict_len=word_dict_len,
        label_dict_len=label_dict_len,
        is_train=False)

    inferer = paddle.inference.Inference(
        output_layer=predict, parameters=parameters)

    test_data = []
    for i, item in enumerate(
            reader.data_reader(test_data_file, word_dict, label_dict)()):
        test_data.append([item[0], item[1]])
        if len(test_data) == batch_size:
            _infer_a_batch(inferer, test_data, word_reverse_dict,
                           label_reverse_dict)
            test_data = []

    _infer_a_batch(inferer, test_data, word_reverse_dict, label_reverse_dict)
    test_data = []


if __name__ == "__main__":
    infer(
        model_path="models/params_pass_0.tar.gz",
        batch_size=2,
        test_data_file="data/test",
        vocab_file="data/vocab.txt",
        target_file="data/target.txt")
