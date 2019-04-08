"""
util tools
"""
from __future__ import print_function
import os
import sys
import numpy as np
import paddle.fluid as fluid


def str2bool(v):
    """
    argparse does not support True or False in python
    """
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    Put arguments to one group
    """
    def __init__(self, parser, title, des):
        """none"""
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """ Add argument """
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args):
    """none"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def to_str(string):
    """none"""
    if isinstance(string, unicode):
        return string.encode("utf-8")
    else:
        return string


def to_lodtensor(data, place):
    """
    Convert data in list into lodtensor.
    """
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


def parse_result(crf_decode, word_list, dataset):
    """ parse result """
    offset_list = (crf_decode.lod())[0]
    np_data = np.array(crf_decode)

    batch_size = len(word_list)
    assert len(offset_list) == batch_size + 1  # assert batch_size

    full_out_str = []
    for sent_index in range(batch_size):
        # assert sentence length
        assert len(word_list[sent_index]) == offset_list[sent_index + 1] - offset_list[sent_index]
        word_index = 0
        outstr = ""
        cur_full_word = ""
        cur_full_tag = ""
        word_index = 0
        for tag_index in range(len(word_list[sent_index])): # iterate every word in sent
            cur_word_id = str(word_list[sent_index][tag_index])
            cur_word = dataset.id2word_dict[cur_word_id]
            cur_label_id = str(np_data[tag_index + offset_list[sent_index]][0])
            cur_tag = dataset.id2label_dict[cur_label_id]
            if cur_tag.endswith("-B") or cur_tag.endswith("O"):
                if len(cur_full_word) != 0:
                    outstr += cur_full_word + u"/" + cur_full_tag + u" "
                cur_full_word = cur_word
                if cur_tag == "O":
                    cur_full_tag = cur_tag
                else:
                    cur_full_tag = cur_tag[0:-2]
            else:
                cur_full_word += cur_word
            word_index += 1
        outstr += cur_full_word + u"/" + cur_full_tag + u" "
        outstr = outstr.strip()
        full_out_str.append(outstr)
    return full_out_str


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """
    Init CheckPoint
    """
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        """
        If existed presitabels
        """
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))


def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program,
                            use_fp16=False):
    """load params of pretrained model, NOT including moment, learning_rate"""
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def _existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=_existed_params)
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))
