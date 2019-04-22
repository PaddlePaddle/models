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
from models.sequence_labeling import nets


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
data_g.add_arg("train_data", str, "./data/train_data", "The folder where the training data is located.")
data_g.add_arg("test_data", str, "./data/test_data", "The folder where the training data is located.")
data_g.add_arg("infer_data", str, "./data/test.tsv", "The folder where the training data is located.")
data_g.add_arg("model_save_dir", str, "./models", "The model will be saved in this path.")
data_g.add_arg("init_checkpoint", str, "", "Path to init model")

data_g.add_arg("corpus_type_list", str, ["human", "feed", "query", "title", "news"],
        "The pattern list of different types of corpus used in training.", nargs='+')
data_g.add_arg("corpus_proportion_list", float, [0.2, 0.2, 0.2, 0.2, 0.2],
        "The proportion list of different types of corpus used in training.", nargs='+')

# 3. train parameters
train_g = utils.ArgumentGroup(parser, "training", "training options")

train_g.add_arg("do_train", bool, True, "whether to perform training")
train_g.add_arg("do_valid", bool, False, "whether to perform validation")
train_g.add_arg("do_test", bool, True, "whether to perform validation")
train_g.add_arg("do_infer", bool, False, "whether to perform inference")
train_g.add_arg("random_seed", int, 0, "random seed for training")
train_g.add_arg("save_model_per_batches", int, 10000, "Save the model once per xxxx batch of training")
train_g.add_arg("valid_model_per_batches", int, 1000, "Do the validation once per xxxx batch of training")
train_g.add_arg("batch_size", int, 80, "The number of sequences contained in a mini-batch, "
        "or the maximum number of tokens (include paddings) contained in a mini-batch.")
train_g.add_arg("epoch", int, 10, "corpus iteration num")
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
            capacity=50,
            shapes=([-1, 1], [-1, 1]),
            dtypes=('int64', 'int64'),
            lod_levels=(1, 1),
            name=pyreader_name,
            use_double_buffer=False)

    words, targets = fluid.layers.read_file(pyreader)
    avg_cost, crf_decode = nets.lex_net(words, targets, args, vocab_size, num_labels)

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
         input=crf_decode,
         label=targets,
         chunk_scheme="IOB",
         num_chunk_types=int(math.ceil((num_labels - 1) / 2.0)))
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    ret = {
        "pyreader":pyreader,
        "words":words,
        "targets":targets,
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
    test_ret["chunk_evaluator"].reset()
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

    # prepare dataset
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
                                dataset.file_reader(args.train_data),
                                buf_size=args.traindata_shuffle_buffer
                            ),
                            batch_size=args.batch_size
                        )
                )

                optimizer = fluid.optimizer.Adam(learning_rate=args.base_learning_rate)
                optimizer.minimize(train_ret["avg_cost"])

    if args.do_test:
        test_program = fluid.Program()
        with fluid.program_guard(test_program, startup_program):
            with fluid.unique_name.guard():
                test_ret = create_model(
                       args, "test_reader", dataset.vocab_size, dataset.num_labels)
                test_ret["pyreader"].decorate_paddle_reader(
                    paddle.batch(
                        dataset.file_reader(args.test_data),
                        batch_size=args.batch_size
                    )
                )
        test_program = test_program.clone(for_test=True)  # to share parameters with train model

    if args.do_infer:
        infer_program = fluid.Program()
        with fluid.program_guard(infer_program, startup_program):
            with fluid.unique_name.guard():
                infer_ret = create_model(
                       args, "infer_reader", dataset.vocab_size, dataset.num_labels)
                infer_ret["pyreader"].decorate_paddle_reader(
                    paddle.batch(
                        dataset.file_reader(args.infer_data),
                        batch_size=args.batch_size
                    )
                )
        infer_program = infer_program.clone(for_test=True)


    # init executor
    if args.use_gpu >= 0:
        place = fluid.CUDAPlace(args.use_gpu)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = multiprocessing.cpu_count()
    exe = fluid.Executor(place)
    exe.run(startup_program)

    # load checkpoints
    if args.do_train:
        if args.init_checkpoint:
            utils.init_checkpoint(exe, args.init_checkpoint, train_program)
    elif args.do_test:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if only doing validation or testing!")
        utils.init_checkpoint(exe, args.init_checkpoint, test_program)
    if args.do_infer:
        utils.init_checkpoint(exe, args.init_checkpoint, infer_program)

    # do start to train
    if args.do_train:
        num_train_examples = dataset.get_num_examples(args.train_data)
        max_train_steps = args.epoch * num_train_examples // args.batch_size
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        batch_id = 0
        for epoch_id in range(args.epoch):
            train_ret["pyreader"].start()
            try:
                while True:
                    start_time = time.time()
                    avg_cost, nums_infer, nums_label, nums_correct = exe.run(
                        train_program,
                        fetch_list=[
                            train_ret["avg_cost"],
                            train_ret["num_infer_chunks"],
                            train_ret["num_label_chunks"],
                            train_ret["num_correct_chunks"],
                        ],
                    )
                    end_time = time.time()
                    train_ret["chunk_evaluator"].reset()
                    train_ret["chunk_evaluator"].update(nums_infer, nums_label, nums_correct)
                    precision, recall, f1_score = train_ret["chunk_evaluator"].eval()
                    batch_id += 1
                    print("[train] batch_id = %d, loss = %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time %.5f " % (
                        batch_id, avg_cost, precision, recall, f1_score, end_time - start_time))

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
        infer_ret["pyreader"].start()
        while True:
            try:
                (words, crf_decode, ) = exe.run(infer_program,
                        fetch_list=[
                            infer_ret["words"],
                            infer_ret["crf_decode"],
                        ],
                        return_numpy=False)
                results = utils.parse_result(words, crf_decode, dataset)
                for result in results:
                    print(result)
            except fluid.core.EOFException:
                infer_ret["pyreader"].reset()
                break


if __name__ == "__main__":
    main(args)
