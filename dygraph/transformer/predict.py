# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import ast

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.dataset.wmt16 as wmt16

from model import TransFormer
from config import *
from data_util import *


def parse_args():
    parser = argparse.ArgumentParser("Arguments for Inference")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument(
        "--model_file",
        type=str,
        default="transformer_params",
        help="Load model from the file named `model_file.pdparams`.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="predict.txt",
        help="The file to output the translation results of predict_file to.")
    parser.add_argument('opts',
                        help='See config.py for all options',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    merge_cfg_from_list(args.opts, [InferTaskConfig, ModelHyperParams])
    return args


def prepare_infer_input(insts, src_pad_idx, bos_idx, n_head):
    """
    inputs for inferencs
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    # start tokens
    trg_word = np.asarray([[bos_idx]] * len(insts), dtype="int64")
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, 1, 1]).astype("float32")
    trg_word = trg_word.reshape(-1, 1, 1)
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)

    data_inputs = [
        src_word, src_pos, src_slf_attn_bias, trg_word, trg_src_attn_bias
    ]

    var_inputs = []
    for i, field in enumerate(encoder_data_input_fields +
                              fast_decoder_data_input_fields):
        var_inputs.append(to_variable(data_inputs[i], name=field))

    enc_inputs = var_inputs[0:len(encoder_data_input_fields)]
    dec_inputs = var_inputs[len(encoder_data_input_fields):]
    return enc_inputs, dec_inputs


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def infer(args):
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        transformer = TransFormer(
            'transformer', ModelHyperParams.src_vocab_size,
            ModelHyperParams.trg_vocab_size, ModelHyperParams.max_length + 1,
            ModelHyperParams.n_layer, ModelHyperParams.n_head,
            ModelHyperParams.d_key, ModelHyperParams.d_value,
            ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
            ModelHyperParams.prepostprocess_dropout,
            ModelHyperParams.attention_dropout, ModelHyperParams.relu_dropout,
            ModelHyperParams.preprocess_cmd, ModelHyperParams.postprocess_cmd,
            ModelHyperParams.weight_sharing)
        # load checkpoint
        model_dict, _ = fluid.load_dygraph(args.model_file)
        transformer.load_dict(model_dict)
        print("checkpoint loaded")
        # start evaluate mode
        transformer.eval()

        reader = paddle.batch(wmt16.test(ModelHyperParams.src_vocab_size,
                                         ModelHyperParams.trg_vocab_size),
                              batch_size=InferTaskConfig.batch_size)
        id2word = wmt16.get_dict("de",
                                 ModelHyperParams.trg_vocab_size,
                                 reverse=True)

        f = open(args.output_file, "wb")
        for batch in reader():
            enc_inputs, dec_inputs = prepare_infer_input(
                batch, ModelHyperParams.eos_idx, ModelHyperParams.bos_idx,
                ModelHyperParams.n_head)

            finished_seq, finished_scores = transformer.beam_search(
                enc_inputs,
                dec_inputs,
                bos_id=ModelHyperParams.bos_idx,
                eos_id=ModelHyperParams.eos_idx,
                max_len=InferTaskConfig.max_out_len,
                alpha=InferTaskConfig.alpha)
            finished_seq = finished_seq.numpy()
            finished_scores = finished_scores.numpy()
            for ins in finished_seq:
                for beam in ins:
                    id_list = post_process_seq(beam, ModelHyperParams.bos_idx,
                                                ModelHyperParams.eos_idx)
                    word_list = [id2word[id] for id in id_list]
                    sequence = " ".join(word_list) + "\n"
                    f.write(sequence.encode("utf8"))
                    break  # only print the best


if __name__ == '__main__':
    args = parse_args()
    infer(args)
