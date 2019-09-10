#coding:utf8
import argparse
import sys

import paddle.fluid as fluid
import os

import creator
import reader
import utils
from reader import load_kv_dict
sys.path.append('../models/')
from model_check import check_cuda

parser = argparse.ArgumentParser(__doc__)

# 1. model parameters
model_g = utils.ArgumentGroup(parser, "model", "model configuration")
model_g.add_arg("word_emb_dim", int, 128, "The dimension in which a word is embedded.")
model_g.add_arg("grnn_hidden_dim", int, 256, "The number of hidden nodes in the GRNN layer.")
model_g.add_arg("bigru_num", int, 2, "The number of bi_gru layers in the network.")
model_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")

# 2. data parameters
data_g = utils.ArgumentGroup(parser, "data", "data paths")
data_g.add_arg("word_dict_path", str, "./conf/word.dic", "The path of the word dictionary.")
data_g.add_arg("label_dict_path", str, "./conf/tag.dic", "The path of the label dictionary.")
data_g.add_arg("word_rep_dict_path", str, "./conf/q2b.dic", "The path of the word replacement Dictionary.")
data_g.add_arg("infer_data", str, "./data/infer.tsv", "The folder where the training data is located.")
data_g.add_arg("init_checkpoint", str, "", "Path to init model")
data_g.add_arg("inference_save_dir", str, "inference_model", "Path to save inference model")

def save_inference_model(args):

    # model definition
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    dataset = reader.Dataset(args)
    infer_program = fluid.Program()
    with fluid.program_guard(infer_program, fluid.default_startup_program()):
        with fluid.unique_name.guard():

            infer_ret = creator.create_model(
                args, dataset.vocab_size, dataset.num_labels, mode='infer')
            infer_program = infer_program.clone(for_test=True)


    # load pretrain check point
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    utils.init_checkpoint(exe, args.init_checkpoint+'.pdckpt', infer_program)

    fluid.io.save_inference_model(args.inference_save_dir,
                                  ['words'],
                                  infer_ret['crf_decode'],
                                  exe,
                                  main_program=infer_program,
                                  model_filename='model.pdmodel',
                                  params_filename='params.pdparams')


def test_inference_model(model_dir, text_list, dataset):
    """
    :param model_dir: model's dir
    :param text_list: a list of input text, which decode as unicode
    :param dataset:
    :return:
    """
    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # transfer text data to input tensor
    UNK = dataset.word2id_dict[u'OOV']
    lod = []
    for text in text_list:
        lod.append([dataset.word2id_dict.get(word, UNK) for word in text])
    base_shape = [[len(c) for c in lod]]
    tensor_words = fluid.create_lod_tensor(lod, base_shape, place)

    # load inference model
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inferencer, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_dir, exe,
                 model_filename='model.pdmodel',
                 params_filename='params.pdparams')
        assert feed_target_names[0] == "words"
        print("Load inference model from %s"%(model_dir))

        # get lac result
        crf_decode = exe.run(inferencer,
                             feed={feed_target_names[0]:tensor_words},
                             fetch_list=fetch_targets,
                             return_numpy=False)

        # parse the crf_decode result
        result = utils.parse_result(tensor_words,crf_decode[0], dataset)
        for i,(sent, tags) in enumerate(result):
            result_list = ['(%s, %s)'%(ch, tag) for ch, tag in zip(sent,tags)]
            print(''.join(result_list))


if __name__=="__main__":
    parser = argparse.ArgumentParser(__doc__)
    utils.load_yaml(parser,'conf/args.yaml')
    args = parser.parse_args()
    check_cuda(args.use_cuda)
    print("save inference model")
    save_inference_model(args)
    
    print("inference model save in %s.pdmodel"%args.inference_save_dir)
    print("test inference model")
    word_dict = load_kv_dict(args.word_dict_path, reverse=True, value_func=int)
    dataset = reader.Dataset(args)
    test_data = [u'百度是一家高科技公司', u'中山大学是岭南第一学府']
    test_inference_model(args.inference_save_dir, test_data, dataset)
