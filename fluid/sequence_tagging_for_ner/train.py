import os
import math
import numpy as np

import paddle.v2 as paddle
import paddle.fluid as fluid

import reader
from network_conf import ner_net
from utils import logger, load_dict
from utils_extend import to_lodtensor, get_embedding


def test(exe, chunk_evaluator, inference_program, test_data, place):
    chunk_evaluator.reset(exe)
    for data in test_data():
        word = to_lodtensor(map(lambda x: x[0], data), place)
        mark = to_lodtensor(map(lambda x: x[1], data), place)
        target = to_lodtensor(map(lambda x: x[2], data), place)
        acc = exe.run(inference_program,
                      feed={"word": word,
                            "mark": mark,
                            "target": target})
    return chunk_evaluator.eval(exe)


def main(train_data_file, test_data_file, vocab_file, target_file, emb_file,
         model_save_dir, num_passes, use_gpu, parallel):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    BATCH_SIZE = 200
    word_dict = load_dict(vocab_file)
    label_dict = load_dict(target_file)

    word_vector_values = get_embedding(emb_file)

    word_dict_len = len(word_dict)
    label_dict_len = len(label_dict)

    avg_cost, feature_out, word, mark, target = ner_net(
        word_dict_len, label_dict_len, parallel)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd_optimizer.minimize(avg_cost)

    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    chunk_evaluator = fluid.evaluator.ChunkEvaluator(
        input=crf_decode,
        label=target,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_target = chunk_evaluator.metrics + chunk_evaluator.states
        inference_program = fluid.io.get_inference_program(test_target)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.data_reader(train_data_file, word_dict, label_dict),
            buf_size=20000),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.data_reader(test_data_file, word_dict, label_dict),
            buf_size=20000),
        batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[word, mark, target], place=place)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    embedding_name = 'emb'
    embedding_param = fluid.global_scope().find_var(embedding_name).get_tensor()
    embedding_param.set(word_vector_values, place)

    batch_id = 0
    for pass_id in xrange(num_passes):
        chunk_evaluator.reset(exe)
        for data in train_reader():
            cost, batch_precision, batch_recall, batch_f1_score = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost] + chunk_evaluator.metrics)
            if batch_id % 5 == 0:
                print("Pass " + str(pass_id) + ", Batch " + str(
                    batch_id) + ", Cost " + str(cost[0]) + ", Precision " + str(
                        batch_precision[0]) + ", Recall " + str(batch_recall[0])
                      + ", F1_score" + str(batch_f1_score[0]))
            batch_id = batch_id + 1

        pass_precision, pass_recall, pass_f1_score = chunk_evaluator.eval(exe)
        print("[TrainSet] pass_id:" + str(pass_id) + " pass_precision:" + str(
            pass_precision) + " pass_recall:" + str(pass_recall) +
              " pass_f1_score:" + str(pass_f1_score))
        pass_precision, pass_recall, pass_f1_score = test(
            exe, chunk_evaluator, inference_program, test_reader, place)
        print("[TestSet] pass_id:" + str(pass_id) + " pass_precision:" + str(
            pass_precision) + " pass_recall:" + str(pass_recall) +
              " pass_f1_score:" + str(pass_f1_score))

        save_dirname = os.path.join(model_save_dir, "params_pass_%d" % pass_id)
        fluid.io.save_inference_model(save_dirname, ['word', 'mark', 'target'],
                                      [crf_decode], exe)


if __name__ == "__main__":
    main(
        train_data_file="data/train",
        test_data_file="data/test",
        vocab_file="data/vocab.txt",
        target_file="data/target.txt",
        emb_file="data/wordVectors.txt",
        model_save_dir="models",
        num_passes=1000,
        use_gpu=False,
        parallel=False)
