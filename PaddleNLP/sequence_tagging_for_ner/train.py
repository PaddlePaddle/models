from __future__ import print_function

import os
import math
import time
import numpy as np
import six

import paddle
import paddle.fluid as fluid

import reader
from network_conf import ner_net
from utils import logger, load_dict
from utils_extend import to_lodtensor, get_embedding


def test(exe, chunk_evaluator, inference_program, test_data, test_fetch_list,
         place):
    chunk_evaluator.reset()
    for data in test_data():
        word = to_lodtensor([x[0] for x in data], place)
        mark = to_lodtensor([x[1] for x in data], place)
        target = to_lodtensor([x[2] for x in data], place)
        rets = exe.run(inference_program,
                       feed={"word": word,
                             "mark": mark,
                             "target": target},
                       fetch_list=test_fetch_list)
        num_infer = np.array(rets[0])
        num_label = np.array(rets[1])
        num_correct = np.array(rets[2])
        chunk_evaluator.update(num_infer[0].astype('int64'),
                               num_label[0].astype('int64'),
                               num_correct[0].astype('int64'))
    return chunk_evaluator.eval()


def main(train_data_file,
         test_data_file,
         vocab_file,
         target_file,
         emb_file,
         model_save_dir,
         num_passes,
         use_gpu,
         parallel,
         batch_size=200):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    word_dict = load_dict(vocab_file)
    label_dict = load_dict(target_file)

    word_vector_values = get_embedding(emb_file)

    word_dict_len = len(word_dict)
    label_dict_len = len(label_dict)

    if "CE_MODE_X" in os.environ:
        fluid.default_startup_program().random_seed = 110

    avg_cost, feature_out, word, mark, target = ner_net(
        word_dict_len, label_dict_len, parallel)

    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
         input=crf_decode,
         label=target,
         chunk_scheme="IOB",
         num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))
    chunk_evaluator = fluid.metrics.ChunkEvaluator()

    inference_program = fluid.default_main_program().clone(for_test=True)
    test_fetch_list = [num_infer_chunks, num_label_chunks, num_correct_chunks]
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd_optimizer.minimize(avg_cost)

    if "CE_MODE_X" not in os.environ:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.data_reader(train_data_file, word_dict, label_dict),
                buf_size=20000),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.data_reader(test_data_file, word_dict, label_dict),
                buf_size=20000),
            batch_size=batch_size)
    else:
        train_reader = paddle.batch(
            reader.data_reader(train_data_file, word_dict, label_dict),
            batch_size=batch_size)
        test_reader = paddle.batch(
            reader.data_reader(test_data_file, word_dict, label_dict),
            batch_size=batch_size)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[word, mark, target], place=place)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    embedding_name = 'emb'
    embedding_param = fluid.global_scope().find_var(embedding_name).get_tensor()
    embedding_param.set(word_vector_values, place)

    time_begin = time.time()
    for pass_id in six.moves.xrange(num_passes):
        chunk_evaluator.reset()
        for batch_id, data in enumerate(train_reader()):
            cost_var, nums_infer, nums_label, nums_correct = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[
                    avg_cost, num_infer_chunks, num_label_chunks,
                    num_correct_chunks
                ])
            if batch_id % 5 == 0:
                print("Pass " + str(pass_id) + ", Batch " + str(batch_id) +
                      ", Cost " + str(cost_var[0]))
            chunk_evaluator.update(nums_infer, nums_label, nums_correct)
        pass_precision, pass_recall, pass_f1_score = chunk_evaluator.eval()
        print("[TrainSet] pass_id:" + str(pass_id) + " pass_precision:" + str(
            pass_precision) + " pass_recall:" + str(pass_recall) +
              " pass_f1_score:" + str(pass_f1_score))

        test_pass_precision, test_pass_recall, test_pass_f1_score = test(
            exe, chunk_evaluator, inference_program, test_reader,
            test_fetch_list, place)
        print("[TestSet] pass_id:" + str(pass_id) + " pass_precision:" + str(
            test_pass_precision) + " pass_recall:" + str(test_pass_recall) +
              " pass_f1_score:" + str(test_pass_f1_score))

        save_dirname = os.path.join(model_save_dir, "params_pass_%d" % pass_id)
        if "CE_MODE_X" not in os.environ:
            fluid.io.save_inference_model(save_dirname, ['word', 'mark'],
                                          crf_decode, exe)

    if "CE_MODE_X" in os.environ:
        print("kpis	train_precision	%f" % pass_precision)
        print("kpis	test_precision	%f" % test_pass_precision)
        print("kpis	train_duration	%f" % (time.time() - time_begin))


if __name__ == "__main__":
    main(
        train_data_file="data/train",
        test_data_file="data/test",
        vocab_file="data/vocab.txt",
        target_file="data/target.txt",
        emb_file="data/wordVectors.txt",
        model_save_dir="models",
        num_passes=2000,
        use_gpu=False,
        parallel=False)
