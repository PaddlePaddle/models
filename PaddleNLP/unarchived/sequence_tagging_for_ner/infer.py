from __future__ import print_function

import numpy as np
import six

import paddle
import paddle.fluid as fluid

from network_conf import ner_net
import reader
from utils import load_dict, load_reverse_dict
from utils_extend import to_lodtensor


def infer(model_path, batch_size, test_data_file, vocab_file, target_file,
          use_gpu):
    """
    use the model under model_path to predict the test data, the result will be printed on the screen

    return nothing
    """
    word_dict = load_dict(vocab_file)
    word_reverse_dict = load_reverse_dict(vocab_file)

    label_dict = load_dict(target_file)
    label_reverse_dict = load_reverse_dict(target_file)

    test_data = paddle.batch(
        reader.data_reader(test_data_file, word_dict, label_dict),
        batch_size=batch_size)
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        for data in test_data():
            word = to_lodtensor([x[0] for x in data], place)
            mark = to_lodtensor([x[1] for x in data], place)
            crf_decode = exe.run(inference_program,
                                 feed={"word": word,
                                       "mark": mark},
                                 fetch_list=fetch_targets,
                                 return_numpy=False)
            lod_info = (crf_decode[0].lod())[0]
            np_data = np.array(crf_decode[0])
            assert len(data) == len(lod_info) - 1
            for sen_index in six.moves.xrange(len(data)):
                assert len(data[sen_index][0]) == lod_info[
                    sen_index + 1] - lod_info[sen_index]
                word_index = 0
                for tag_index in six.moves.xrange(lod_info[sen_index],
                                                  lod_info[sen_index + 1]):
                    word = word_reverse_dict[data[sen_index][0][word_index]]
                    gold_tag = label_reverse_dict[data[sen_index][2][
                        word_index]]
                    tag = label_reverse_dict[np_data[tag_index][0]]
                    print(word + "\t" + gold_tag + "\t" + tag)
                    word_index += 1
                print("")


if __name__ == "__main__":
    infer(
        model_path="models/params_pass_0",
        batch_size=6,
        test_data_file="data/test",
        vocab_file="data/vocab.txt",
        target_file="data/target.txt",
        use_gpu=False)
