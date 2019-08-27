# coding:utf8
"""
util tools
"""

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
from  eval import test_process

import nets
import sys
import warnings
sys.path.append('../models/')
from model_check import check_cuda
import ipdb

# yapf: disable
# parser = argparse.ArgumentParser(__doc__)

# 1. model parameters
# model_g = utils.ArgumentGroup(parser, "model", "model configuration")
# model_g.add_arg("word_emb_dim", int, 128, "The dimension in which a word is embedded.")
# model_g.add_arg("grnn_hidden_dim", int, 256, "The number of hidden nodes in the GRNN layer.")
# model_g.add_arg("bigru_num", int, 2, "The number of bi_gru layers in the network.")
#
# # 2. data parameters
# data_g = utils.ArgumentGroup(parser, "data", "data paths")
# data_g.add_arg("word_dict_path", str, "./conf/word.dic", "The path of the word dictionary.")
# data_g.add_arg("label_dict_path", str, "./conf/tag.dic", "The path of the label dictionary.")
# data_g.add_arg("word_rep_dict_path", str, "./conf/q2b.dic", "The path of the word replacement Dictionary.")
# data_g.add_arg("train_data", str, "./data/train.tsv", "The folder where the train data is located.")
# data_g.add_arg("test_data", str, "./data/test.tsv", "The folder where the test data is located.")
# data_g.add_arg("infer_data", str, "./data/test.tsv", "The folder where the infer data is located.")
# data_g.add_arg("model_save_dir", str, "./models", "The model will be saved in this path.")
# data_g.add_arg("init_checkpoint", str, "", "Path to init model")
#
# # 3. train parameters
# train_g = utils.ArgumentGroup(parser, "training", "training options")
# train_g.add_arg("random_seed", int, 0, "random seed for training")
# train_g.add_arg("print_step", int, 200, "print the result per xxx batch of training")
# train_g.add_arg("save_model_per_batches", int, 10000, "Save the model once per xxxx batch of training")
# train_g.add_arg("valid_model_per_batches", int, 1000, "Do the validation once per xxxx batch of training")
# train_g.add_arg("batch_size", int, 80, "The number of sequences contained in a mini-batch, "
#         "or the maximum number of tokens (include paddings) contained in a mini-batch.")
# train_g.add_arg("epoch", int, 10, "corpus iteration num")
# train_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")
#train_g.add_arg("cpu_num", int, 2, "The number of cpu used to train model",
#         "it works when use_cuda=False)
# train_g.add_arg("traindata_shuffle_buffer", int, 2000, "The buffer size used in shuffle the training data.")
# train_g.add_arg("base_learning_rate", float, 1e-3, "The basic learning rate that affects the entire network.")
# train_g.add_arg("emb_learning_rate", float, 5,
#     "The real learning rate of the embedding layer will be (emb_learning_rate * base_learning_rate).")
# train_g.add_arg("crf_learning_rate", float, 0.2,
#     "The real learning rate of the embedding layer will be (crf_learning_rate * base_learning_rate).")
#
# parser.add_argument('--enable_ce', action='store_true', help='If set, run the task with continuous evaluation logs.')


# yapf: enable.

# the function to train model
def do_train(args):
    train_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    dataset = reader.Dataset(args)
    with fluid.program_guard(train_program, startup_program):
        train_program.random_seed = args.random_seed
        startup_program.random_seed = args.random_seed

        with fluid.unique_name.guard():
            train_ret = nets.create_model(
                args, dataset.vocab_size, dataset.num_labels, mode='train')
            test_program = train_program.clone(for_test=True)

            optimizer = fluid.optimizer.Adam(learning_rate=args.base_learning_rate)
            optimizer.minimize(train_ret["avg_cost"])


    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        dev_count = min(multiprocessing.cpu_count(), args.cpu_num)
        if(dev_count<args.cpu_num):
            warnings.warn('The total CPU NUM in this machine is %d, which is less than cpu_num parameter you set'%dev_count)
            warnings.warn('Change the cpu_num from %d to %d'%(args.cpu_num, dev_count))
        os.environ['CPU_NUM'] = str(dev_count)
        place = fluid.CPUPlace()


    # init reader
    train_reader = fluid.io.PyReader(
        feed_list=[train_ret['words'], train_ret['targets']],
        capacity=300,
        use_double_buffer=True,
        iterable=True
    )
    train_reader.decorate_sample_list_generator(
        paddle.batch(
            paddle.reader.shuffle(
                dataset.file_reader(args.train_data),
                buf_size=args.traindata_shuffle_buffer
            ),
            batch_size=args.batch_size
        ),
        places=place
    )


    test_reader = fluid.io.PyReader(
        feed_list=[train_ret['words'], train_ret['targets']],
        capacity=300,
        use_double_buffer=True,
        iterable=True
    )
    test_reader.decorate_sample_list_generator(
        paddle.batch(
            paddle.reader.shuffle(
                dataset.file_reader(args.test_data),
                buf_size=args.traindata_shuffle_buffer
            ),
            batch_size=args.batch_size
        ),
        places=place
    )

    exe = fluid.Executor(place)
    exe.run(startup_program)

    if args.init_checkpoint:
        utils.init_checkpoint(exe, args.init_checkpoint, train_program)
    if dev_count>1:
        device = "GPU" if args.use_cuda else "CPU"
        print("%d %s are used to train model"%(dev_count, device))
        # multi cpu/gpu config
        exec_strategy = fluid.ExecutionStrategy()
        # exec_strategy.num_threads = dev_count * 6
        build_strategy = fluid.compiler.BuildStrategy()
        # build_strategy.enable_inplace = True

        compiled_prog = fluid.compiler.CompiledProgram(train_program).with_data_parallel(
            loss_name=train_ret['avg_cost'].name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy
        )
    else:
        compiled_prog = fluid.compiler.CompiledProgram(train_program)

    # start training
    num_train_examples = dataset.get_num_examples(args.train_data)
    max_train_steps = args.epoch * num_train_examples // args.batch_size
    print("Num train examples: %d" % num_train_examples)
    print("Max train steps: %d" % max_train_steps)

    ce_info = []
    step = 0
    for epoch_id in range(args.epoch):
        # pyreader.start()
        ce_time = 0
        try:
            # while True:
            # ipdb.set_trace()
            for data in train_reader():
                # this is for minimizing the fetching op, saving the training speed.
                if step % args.print_step == 0:
                    fetch_list = [
                        train_ret["avg_cost"],
                        train_ret["precision"],
                        train_ret["recall"],
                        train_ret["f1_score"]
                    ]
                else:
                    fetch_list = []

                start_time = time.time()

                output = exe.run(
                    compiled_prog,
                    # train_program,
                    fetch_list=fetch_list,
                    feed=data[0],
                )

                end_time = time.time()
                if step % args.print_step == 0:
                    avg_cost, precision, recall, f1_score = [np.mean(x) for x in output]

                    print("[train] step = %d, loss = %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time %.5f"%(
                        step, avg_cost, precision, recall, f1_score, end_time-start_time))
                    test_process(exe, test_program, test_reader, train_ret)

                    ce_time += end_time - start_time
                    ce_info.append([ce_time, avg_cost, precision, recall, f1_score])


                # save checkpoints
                if step % args.save_model_per_batches == 0 and step != 0:
                    save_path = os.path.join(args.model_save_dir, "step_" + str(step)+'.pdckpt')
                    fluid.io.save_persistables(exe, save_path, train_program)
                step += 1
        except fluid.core.EOFException:
            save_path = os.path.join(args.model_save_dir, "step_" + str(step))
            fluid.io.save_persistables(exe, save_path, train_program)


    if args.enable_ce:
        card_num = get_cards()
        ce_cost = 0
        ce_f1 = 0
        ce_p = 0
        ce_r = 0
        ce_time = 0
        try:
            ce_time = ce_info[-2][0]
            ce_cost = ce_info[-2][1]
            ce_p = ce_info[-2][2]
            ce_r = ce_info[-2][3]
            ce_f1 = ce_info[-2][4]
        except:
            print("ce info error")
        print("kpis\teach_step_duration_card%s\t%s" %
                (card_num, ce_time))
        print("kpis\ttrain_cost_card%s\t%f" %
            (card_num, ce_cost))
        print("kpis\ttrain_precision_card%s\t%f" %
            (card_num, ce_p))
        print("kpis\ttrain_recall_card%s\t%f" %
            (card_num, ce_r))
        print("kpis\ttrain_f1_card%s\t%f" %
            (card_num, ce_f1))


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num

if __name__ == "__main__":
    # 参数控制可以根据需求使用argparse，yaml或者json
    # 对NLP任务推荐使用PALM下定义的configure，可以统一argparse，yaml或者json格式的配置文件。

    parser = argparse.ArgumentParser(__doc__)
    utils.load_yaml(parser,'conf/args.yaml')

    args = parser.parse_args()
    check_cuda(args.use_cuda)

    print(args)

    do_train(args)

