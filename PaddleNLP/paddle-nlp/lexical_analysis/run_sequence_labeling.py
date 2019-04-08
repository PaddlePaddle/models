"""
Baidu's open-source Lexical Analysis tool for Chinese, including:
    1. Word Segmentation,
    2. Part-of-Speech Tagging
    3. Named Entity Recognition
"""

from __future__ import print_function

import os
import sys
import math
import time
import random
import argparse
import multiprocessing

import numpy as np
import paddle
import paddle.fluid as fluid

import reader
import utils

sys.path.append("../")
from models.seq_lab import nets


# yapf: disable
parser = argparse.ArgumentParser(__doc__)

# 1. model parameters
model_g = utils.ArgumentGroup(parser, "model", "model configuration")
model_g.add_arg("word_emb_dim", int, 128, "The dimension in which a word is embedded.")
model_g.add_arg("grnn_hidden_dim", int, 256, "The number of hidden nodes in the GRNN layer.")
model_g.add_arg("bigru_num", int, 2, "The number of bi_gru layers in the network.")

# 2. data parameters
data_g = utils.ArgumentGroup(parser, "data", "data paths")
data_g.add_arg("word_dict_path", str, "./conf/word.dic", "The path of the word dictionary.")
data_g.add_arg("label_dict_path", str, "./conf/tag.dic", "The path of the label dictionary.")
data_g.add_arg("word_rep_dict_path", str, "./conf/q2b.dic", "The path of the word replacement Dictionary.")
data_g.add_arg("traindata_dir", str, "./data/train_data", "The folder where the training data is located.")
data_g.add_arg("testdata_dir", str, "./data/test_data", "The folder where the training data is located.")
data_g.add_arg("model_save_dir", str, "./models", "The model will be saved in this path.")
data_g.add_arg("model_path", str, "./model/", "Path to load the model for inference")

data_g.add_arg("corpus_type_list", str, ["human", "feed", "query", "title", "news"],
        "The pattern list of different types of corpus used in training.", nargs='+')
data_g.add_arg("corpus_proportion_list", float, [0.2, 0.2, 0.2, 0.2, 0.2],
        "The proportion list of different types of corpus used in training.", nargs='+')

# 3. train parameters
train_g = utils.ArgumentGroup(parser, "training", "training options")

train_g.add_arg("do_train", bool, True, "whether to perform training")
train_g.add_arg("do_valid", bool, True, "whether to perform validation")
train_g.add_arg("do_test", bool, True, "whether to perform validation")
train_g.add_arg("do_infer", bool, True, "whether to perform inference")
train_g.add_arg("random_seed", int, 0, "random seed for training")
train_g.add_arg("num_iterations", int, 0,
        "The maximum number of iterations. If set to 0 (default), do not limit the number.")
train_g.add_arg("save_model_per_batches", int, 10, "Save the model once per xxxx batch of training")
train_g.add_arg("valid_model_per_batches", int, 10, "Do the validation once per xxxx batch of training")
train_g.add_arg("eval_window", int, 20, "abandoned, Training will be suspended when the evaluation indicators " \
        "on the validation set no longer increase. The eval_window specifies the scope of the evaluation.")
train_g.add_arg("batch_size", int, 80, "The number of sequences contained in a mini-batch, "
        "or the maximum number of tokens (include paddings) contained in a mini-batch.")
train_g.add_arg("corpus_num", int, 10, "corpus iteration num")
train_g.add_arg("use_gpu", int, -1, "Whether or not to use GPU. -1 means CPU, else GPU id")
train_g.add_arg("traindata_shuffle_buffer", int, 200, "The buffer size used in shuffle the training data.")
train_g.add_arg("base_learning_rate", float, 1e-3, "The basic learning rate that affects the entire network.")
train_g.add_arg("emb_learning_rate", float, 5,
    "The real learning rate of the embedding layer will be (emb_learning_rate * base_learning_rate).")
train_g.add_arg("crf_learning_rate", float, 0.2,
    "The real learning rate of the embedding layer will be (crf_learning_rate * base_learning_rate).")


args = parser.parse_args()
# yapf: enable.

if len(args.corpus_proportion_list) != len(args.corpus_type_list):
    sys.stderr.write(
        "The length of corpus_proportion_list should be equal to the length of corpus_type_list.\n"
    )
    exit(-1)

print(args)


def create_model(args, pyreader_name, vocab_size, num_labels):
    """create lac model"""
    pyreader = fluid.layers.py_reader(
            capacity=16,
            shapes=([-1, 1], [-1, 1]),
            dtypes=('int64', 'int64'),
            lod_levels=(1, 1),
            name=pyreader_name,
            use_double_buffer=False)

    word, target = fluid.layers.read_file(pyreader)
    avg_cost, crf_decode = nets.lex_net(word, target, args, vocab_size, num_labels)

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
         input=crf_decode,
         label=target,
         chunk_scheme="IOB",
         num_chunk_types=int(math.ceil((num_labels - 1) / 2.0)))
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    ret = {
        "pyreader":pyreader,
        "avg_cost":avg_cost,
        "crf_decode":crf_decode,
        "chunk_evaluator":chunk_evaluator,
        "num_infer_chunks":num_infer_chunks,
        "num_label_chunks":num_label_chunks,
        "num_correct_chunks":num_correct_chunks
    }

    return ret


def evaluate(exe, test_program, test_ret):
    """evaluate for test data"""
    test_ret["pyreader"].start()
    loss = []
    precision = []
    recall = []
    f1 = []
    start_time = time.time()
    while True:
        try:
            avg_loss, nums_infer, nums_label, nums_correct = exe.run(
                    test_program,
                    fetch_list=[
                        test_ret["avg_cost"],
                        test_ret["num_infer_chunks"],
                        test_ret["num_label_chunks"],
                        test_ret["num_correct_chunks"],
                    ],
            )
            loss.append(avg_loss)

            test_ret["chunk_evaluator"].update(nums_infer, nums_label, nums_correct)
            p, r, f = test_ret["chunk_evaluator"].eval()

            precision.append(p)
            recall.append(r)
            f1.append(f)
        except fluid.core.EOFException:
            test_ret["pyreader"].reset()
            break
    end_time = time.time()
    print("[test] avg loss: %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time: %.3f s"
            % (np.mean(loss), np.mean(precision),
                np.mean(recall), np.mean(f1), end_time - start_time))


def main(args):

    startup_program = fluid.Program()
    if args.random_seed is not None:
        startup_program.random_seed = args.random_seed

    # STEP 1. prepare dataset
    dataset = reader.Dataset(args)

    if args.do_train:
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            with fluid.unique_name.guard():
                train_ret = create_model(
                        args, "train_reader", dataset.vocab_size, dataset.num_labels)
                train_ret["pyreader"].decorate_paddle_reader(
                        paddle.batch(
                            paddle.reader.shuffle(
                                dataset.file_reader(args.traindata_dir),
                                buf_size=args.traindata_shuffle_buffer
                            ),
                            batch_size=args.batch_size
                        )
                )

                sgd_optimizer = fluid.optimizer.SGD(learning_rate=args.base_learning_rate)
                sgd_optimizer.minimize(train_ret["avg_cost"])

    if args.do_test:
        test_program = fluid.Program()
        with fluid.program_guard(test_program, startup_program):
            with fluid.unique_name.guard():
                test_ret = create_model(
                       args, "test_reader", dataset.vocab_size, dataset.num_labels)
                test_ret["pyreader"].decorate_paddle_reader(
                    paddle.batch(
                        dataset.file_reader(args.testdata_dir),
                        batch_size=args.batch_size
                    )
                )
        test_program = test_program.clone(for_test=True)  # to share parameters with train model


    # STEP 2, run model
    if args.use_gpu >= 0:
        place = fluid.CUDAPlace(args.use_gpu)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = multiprocessing.cpu_count()
    exe = fluid.Executor(place)
    exe.run(startup_program)

    #TODO need codes for continuing training from checkpoint

    if args.do_train:
        batch_id = 0
        for epoch_id in range(args.corpus_num):
            train_ret["pyreader"].start()
            try:
                while True:
                    avg_cost, nums_infer, nums_label, nums_correct = exe.run(
                        train_program,
                        fetch_list=[
                            train_ret["avg_cost"],
                            train_ret["num_infer_chunks"],
                            train_ret["num_label_chunks"],
                            train_ret["num_correct_chunks"],
                        ],
                    )
                    train_ret["chunk_evaluator"].update(nums_infer, nums_label, nums_correct)
                    precision, recall, f1_score = train_ret["chunk_evaluator"].eval()
                    batch_id += 1
                    print("[train] batch_id = %d, loss = %.5f, P: %.5f, R: %.5f, F1: %.5f" % (
                        batch_id, avg_cost, precision, recall, f1_score))

                    # save checkpoints
                    if (batch_id % args.save_model_per_batches == 0):
                        save_path = os.path.join(args.model_save_dir, "step_" + str(batch_id))
                        fluid.io.save_persistables(exe, save_path, train_program)

                    # evaluate
                    if (batch_id % args.valid_model_per_batches == 0) and args.do_test:
                        evaluate(exe, test_program, test_ret)

            except fluid.core.EOFException:
                save_path = os.path.join(args.model_save_dir, "step_" + str(batch_id))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_ret["pyreader"].reset()
                # break?

    # only test
    if args.do_test:
        evaluate(exe, test_program, test_ret)

    if args.do_infer:
        # prepare data
        infer_data = paddle.batch(
            dataset.file_reader(args.testdata_dir),
            batch_size=args.batch_size
        )

        # load checkpoints
        [infer_program, feed_target_names, fetch_targets] = \
                fluid.io.load_inference_model(args.model_path, exe)

        # do infer
        for data in infer_data():
            word_list = [x[0] for x in data]
            word_idx = utils.to_lodtensor(word_list, place)
            (crf_decode, ) = exe.run(infer_program,
                    feed={"word":word_idx}, fetch_list=fetch_targets, return_numpy=False)
            result = utils.parse_result(crf_decode, word_list, dataset)
            print(utils.to_str("\n".join(result)))


if __name__ == "__main__":
    main(args)
