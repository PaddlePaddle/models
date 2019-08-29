#coding:utf8
import argparse
import os
import time
import sys

import paddle.fluid as fluid
import paddle

import utils
import reader
import creator
sys.path.append('../models/')
from model_check import check_cuda

parser = argparse.ArgumentParser(__doc__)
# 1. model parameters
model_g = utils.ArgumentGroup(parser, "model", "model configuration")
model_g.add_arg("word_emb_dim", int, 128, "The dimension in which a word is embedded.")
model_g.add_arg("grnn_hidden_dim", int, 128, "The number of hidden nodes in the GRNN layer.")
model_g.add_arg("bigru_num", int, 2, "The number of bi_gru layers in the network.")
model_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")

# 2. data parameters
data_g = utils.ArgumentGroup(parser, "data", "data paths")
data_g.add_arg("word_dict_path", str, "./conf/word.dic", "The path of the word dictionary.")
data_g.add_arg("label_dict_path", str, "./conf/tag.dic", "The path of the label dictionary.")
data_g.add_arg("word_rep_dict_path", str, "./conf/q2b.dic", "The path of the word replacement Dictionary.")
data_g.add_arg("test_data", str, "./data/test.tsv", "The folder where the training data is located.")
data_g.add_arg("init_checkpoint", str, "./model_baseline", "Path to init model")
data_g.add_arg("batch_size", int, 200, "The number of sequences contained in a mini-batch, "
        "or the maximum number of tokens (include paddings) contained in a mini-batch.")

def do_eval(args):
    dataset = reader.Dataset(args)

    test_program = fluid.Program()
    with fluid.program_guard(test_program, fluid.default_startup_program()):
        with fluid.unique_name.guard():
            test_ret = creator.create_model(
                args, dataset.vocab_size, dataset.num_labels, mode='test')
    test_program = test_program.clone(for_test=True)

    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()

    pyreader = creator.create_pyreader(args, file_name=args.test_data,
                                       feed_list=test_ret['feed_list'],
                                       place=place,
                                       mode='lac',
                                       reader=dataset,
                                       iterable=True,
                                       for_test=True)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load model
    utils.init_checkpoint(exe, args.init_checkpoint+'.pdckpt', test_program)
    test_process(exe=exe,
                 program=test_program,
                 reader=pyreader,
                 test_ret=test_ret
                 )

def test_process(exe, program, reader, test_ret):
    """
    the function to execute the infer process
    :param exe: the fluid Executor
    :param program: the infer_program
    :param reader: data reader
    :return: the list of prediction result
    """
    test_ret["chunk_evaluator"].reset()

    start_time = time.time()
    for data in reader():

        nums_infer, nums_label, nums_correct = exe.run(program,
                                fetch_list=[
                                    test_ret["num_infer_chunks"],
                                    test_ret["num_label_chunks"],
                                    test_ret["num_correct_chunks"],
                                ],
                             feed=data,
                             )

        test_ret["chunk_evaluator"].update(nums_infer, nums_label, nums_correct)
    precision, recall, f1 = test_ret["chunk_evaluator"].eval()
    end_time = time.time()
    print("[test] P: %.5f, R: %.5f, F1: %.5f, elapsed time: %.3f s"
          % (precision, recall, f1, end_time - start_time))

if __name__ == '__main__':
    args = parser.parse_args()
    check_cuda(args.use_cuda)
    do_eval(args)
