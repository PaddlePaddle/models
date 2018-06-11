import os
import math
import time

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer

import reader


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def ner_net(word_dict_len, label_dict_len):
    IS_SPARSE = False
    word_dim = 32
    mention_dict_len = 57
    mention_dim = 20
    grnn_hidden = 36
    emb_lr = 5
    init_bound = 0.1

    def _net_conf(word, mark, target):
        word_embedding = fluid.layers.embedding(
            input=word,
            size=[word_dict_len, word_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))

        mention_embedding = fluid.layers.embedding(
            input=mention,
            size=[mention_dict_len, mention_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="mention_emb",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))

        word_embedding_r = fluid.layers.embedding(
            input=word,
            size=[word_dict_len, word_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="word_emb_r",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))

        mention_embedding_r = fluid.layers.embedding(
            input=mention,
            size=[mention_dict_len, mention_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="mention_emb_r",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))

        word_mention_vector = fluid.layers.concat(
            input=[word_embedding, mention_embedding], axis=1)

        word_mention_vector_r = fluid.layers.concat(
            input=[word_embedding_r, mention_embedding_r], axis=1)

        pre_gru = fluid.layers.fc(
            input=word_mention_vector,
            size=grnn_hidden * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=grnn_hidden,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        pre_gru_r = fluid.layers.fc(
            input=word_mention_vector_r,
            size=grnn_hidden * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=grnn_hidden,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        gru_merged = fluid.layers.concat(input=[gru, gru_r], axis=1)

        emission = fluid.layers.fc(
            size=label_dict_len,
            input=gru_merged,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        crf_cost = fluid.layers.linear_chain_crf(
            input=emission,
            label=target,
            param_attr=fluid.ParamAttr(
                name='crfw',
                learning_rate=0.2, ))
        avg_cost = fluid.layers.mean(x=crf_cost)
        return avg_cost, emission

    word = fluid.layers.data(name='word', shape=[1], dtype='int64', lod_level=1)
    mention = fluid.layers.data(
        name='mention', shape=[1], dtype='int64', lod_level=1)
    target = fluid.layers.data(
        name="target", shape=[1], dtype='int64', lod_level=1)

    avg_cost, emission = _net_conf(word, mention, target)

    return avg_cost, emission, word, mention, target


def test2(exe, chunk_evaluator, inference_program, test_data, place,
          cur_fetch_list):
    chunk_evaluator.reset()
    for data in test_data():
        word = to_lodtensor(map(lambda x: x[0], data), place)
        mention = to_lodtensor(map(lambda x: x[1], data), place)
        target = to_lodtensor(map(lambda x: x[2], data), place)
        result_list = exe.run(
            inference_program,
            feed={"word": word,
                  "mention": mention,
                  "target": target},
            fetch_list=cur_fetch_list)
        number_infer = np.array(result_list[0])
        number_label = np.array(result_list[1])
        number_correct = np.array(result_list[2])
        chunk_evaluator.update(number_infer[0], number_label[0],
                               number_correct[0])
    return chunk_evaluator.eval()


def test(test_exe, chunk_evaluator, inference_program, test_data, place,
         cur_fetch_list):
    chunk_evaluator.reset()
    for data in test_data():
        word = to_lodtensor(map(lambda x: x[0], data), place)
        mention = to_lodtensor(map(lambda x: x[1], data), place)
        target = to_lodtensor(map(lambda x: x[2], data), place)
        result_list = test_exe.run(
            fetch_list=cur_fetch_list,
            feed={"word": word,
                  "mention": mention,
                  "target": target})
        number_infer = np.array(result_list[0])
        number_label = np.array(result_list[1])
        number_correct = np.array(result_list[2])
        chunk_evaluator.update(number_infer.sum(),
                               number_label.sum(), number_correct.sum())
    return chunk_evaluator.eval()


def main(train_data_file, test_data_file, model_save_dir, num_passes):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    BATCH_SIZE = 256
    word_dict_len = 1942563
    label_dict_len = 49

    main = fluid.Program()
    startup = fluid.Program()
    with fluid.program_guard(main, startup):
        avg_cost, feature_out, word, mention, target = ner_net(word_dict_len,
                                                               label_dict_len)

        crf_decode = fluid.layers.crf_decoding(
            input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

        sgd_optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
        sgd_optimizer.minimize(avg_cost)

        (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
         num_correct_chunks) = fluid.layers.chunk_eval(
             input=crf_decode,
             label=target,
             chunk_scheme="IOB",
             num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))

        chunk_evaluator = fluid.metrics.ChunkEvaluator()

        inference_program = fluid.default_main_program().clone()
        with fluid.program_guard(inference_program):
            inference_program = fluid.io.get_inference_program(
                [num_infer_chunks, num_label_chunks, num_correct_chunks])

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.file_reader(train_data_file), buf_size=2000000),
            batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.file_reader(test_data_file), buf_size=2000000),
            batch_size=BATCH_SIZE)

        place = fluid.CUDAPlace(0)
        feeder = fluid.DataFeeder(
            feed_list=[word, mention, target], place=place)

        exe = fluid.Executor(place)

        exe.run(startup)
        train_exe = fluid.ParallelExecutor(
            loss_name=avg_cost.name, use_cuda=True)
        test_exe = fluid.ParallelExecutor(
            use_cuda=True,
            main_program=inference_program,
            share_vars_from=train_exe)
        batch_id = 0
        for pass_id in xrange(num_passes):
            chunk_evaluator.reset()
            train_reader_iter = train_reader()
            start_time = time.time()
            while True:
                try:
                    cur_batch = next(train_reader_iter)
                    cost, nums_infer, nums_label, nums_correct = train_exe.run(
                        fetch_list=[
                            avg_cost.name, num_infer_chunks.name,
                            num_label_chunks.name, num_correct_chunks.name
                        ],
                        feed=feeder.feed(cur_batch))
                    chunk_evaluator.update(
                        np.array(nums_infer).sum(),
                        np.array(nums_label).sum(),
                        np.array(nums_correct).sum())
                    cost_list = np.array(cost)
                    batch_id += 1
                except StopIteration:
                    break
            end_time = time.time()
            print("pass_id:" + str(pass_id) + ", time_cost:" + str(
                end_time - start_time) + "s")
            precision, recall, f1_score = chunk_evaluator.eval()
            print("[Train] precision:" + str(precision) + ", recall:" + str(
                recall) + ", f1:" + str(f1_score))
            p, r, f1 = test2(
                exe, chunk_evaluator, inference_program, test_reader, place,
                [num_infer_chunks, num_label_chunks, num_correct_chunks])
            print("[Test] precision:" + str(p) + ", recall:" + str(r) + ", f1:"
                  + str(f1))
            save_dirname = os.path.join(model_save_dir,
                                        "params_pass_%d" % pass_id)
            fluid.io.save_inference_model(save_dirname, ['word', 'mention'],
                                          [crf_decode], exe)


if __name__ == "__main__":
    main(
        train_data_file="./data/train_files",
        test_data_file="./data/test_files",
        model_save_dir="./output",
        num_passes=1000)
