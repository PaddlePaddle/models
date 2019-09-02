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


class LAC(object):
    """docstring for LAC"""
    def __init__(self, model_path = 'infer_model', use_cuda=False):
        super(LAC, self).__init__()
        check_cuda(use_cuda)

        parser = argparse.ArgumentParser(__doc__)
        utils.load_yaml(parser, 'conf/args.yaml')
        args = parser.parse_args()

        # init executor
        if use_cuda:
            self.place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        else:
            self.place = fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)


        self.dataset = reader.Dataset(args)

        # load inference model
        self.inference_scope = fluid.core.Scope()
        with fluid.scope_guard(self.inference_scope):
            [self.inferencer, self.feed_target_names,
             self.fetch_targets] = fluid.io.load_inference_model(model_path + '.pdmodel', self.exe)
            assert self.feed_target_names[0] == "words"
        print("Load inference model from %s.pdmodel" % (model_path))



    def lac_seg(self, text_list):
#         ipdb.set_trace()
        tensor_words = self.text2tensor(self.dataset,text_list)
        with fluid.scope_guard(self.inference_scope):
            crf_decode = self.exe.run(self.inferencer,
                                 feed={self.feed_target_names[0]: tensor_words},
                                 fetch_list=self.fetch_targets,
                                 return_numpy=False)

        result = utils.parse_result(tensor_words, crf_decode[0], self.dataset)
        return result



    def text2tensor(self, dataset, text_list):
        # transfer text data to input tensor
        UNK = dataset.word2id_dict[u'OOV']
        lod = []
        for text in text_list:
            lod.append([dataset.word2id_dict.get(word, UNK) for word in text])
        base_shape = [[len(c) for c in lod]]
        tensor_words = fluid.create_lod_tensor(lod, base_shape, self.place)

        return tensor_words



if __name__ == "__main__":
    lac = LAC('infer_model')

    test_data = [u'百度是一家高科技公司', u'中山大学是岭南第一学府']
    result = lac.lac_seg(test_data)
    for i, (sent, tags) in enumerate(result):
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print(''.join(result_list))

