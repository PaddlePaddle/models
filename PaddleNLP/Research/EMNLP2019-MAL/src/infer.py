#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import ast
import multiprocessing
import numpy as np
import os
from functools import partial

import contextlib
import time
import paddle.fluid.profiler as profiler

import paddle
import paddle.fluid as fluid

import forward_model
import reader
import sys
from config import *
from forward_model import wrap_encoder as encoder
from forward_model import wrap_decoder as decoder
from forward_model import forward_fast_decode
from dense_model import dense_fast_decode
from relative_model import relative_fast_decode
from forward_model import forward_position_encoding_init
from reader import *


def parse_args():
    """
        parse_args
    """
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        "--val_file_pattern",
        type=str,
        required=True,
        help="The pattern to match test data files.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="The number of examples in one run for sequence generation.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=10000,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--special_token",
        type=str,
        default=["<s>", "<e>", "<unk>"],
        nargs=3,
        help="The <bos>, <eos> and <unk> tokens in the dictionary.")
    parser.add_argument(
        "--token_delimiter",
        type=lambda x: str(x.encode().decode("unicode-escape")),
        default=" ",
        help="The delimiter used to split tokens in source or target sentences. "
        "For EN-DE BPE data we provided, use spaces as token delimiter. ")
    parser.add_argument(
        "--use_mem_opt",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to use memory optimization.")
    parser.add_argument(
        "--use_py_reader",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use py_reader.")
    parser.add_argument(
        "--use_parallel_exe",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use ParallelExecutor.")
    parser.add_argument(
        "--use_candidate",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use candidates.")
    parser.add_argument(
        "--common_ids",
        type=str,
        default="",
        help="The file path of common ids.")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--use_delay_load",
        type=ast.literal_eval,
        default=True,
        help=
        "The flag indicating whether to load all data into memories at once.")
    parser.add_argument(
        "--vocab_size",
        type=str,
        required=True,
        help="Size of Vocab.")
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        help="Infer batch_size")
    parser.add_argument(
        "--decode_alpha",
        type=float,
        help="decode_alpha")

    args = parser.parse_args()
    # Append args related to dict
    #src_dict = reader.DataReader.load_dict(args.src_vocab_fpath)
    #trg_dict = reader.DataReader.load_dict(args.trg_vocab_fpath)
    #dict_args = [
    #    "src_vocab_size", str(len(src_dict)), "trg_vocab_size",
    #    str(len(trg_dict)), "bos_idx", str(src_dict[args.special_token[0]]),
    #    "eos_idx", str(src_dict[args.special_token[1]]), "unk_idx",
    #    str(src_dict[args.special_token[2]])
    #]
    voc_size = args.vocab_size
    dict_args = [
        "src_vocab_size", voc_size,
        "trg_vocab_size", voc_size,
        "bos_idx", str(0),
        "eos_idx", str(1),
        "unk_idx", str(int(voc_size) - 1)
    ]
    merge_cfg_from_list(args.opts + dict_args,
                        [InferTaskConfig, ModelHyperParams])
    return args
    

def post_process_seq(seq,
                     bos_idx=ModelHyperParams.bos_idx,
                     eos_idx=ModelHyperParams.eos_idx,
                     output_bos=InferTaskConfig.output_bos,
                     output_eos=InferTaskConfig.output_eos):
    """
    Post-process the beam-search decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
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


def prepare_batch_input(insts, data_input_names, src_pad_idx, bos_idx, n_head,
                        d_model):
    """
    Put all padded data needed by beam search decoder into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    source_length = np.asarray([src_max_len], dtype="int64")
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)

    data_input_dict = dict(
        zip(data_input_names, [
            src_word, src_pos, src_slf_attn_bias, source_length
        ]))

    return data_input_dict


def prepare_feed_dict_list(data_generator, count):
    """
    Prepare the list of feed dict for multi-devices.
    """
    feed_dict_list = []
    if data_generator is not None:  # use_py_reader == False
        data_input_names = encoder_data_input_fields + fast_decoder_data_input_fields
        data = next(data_generator)
        for idx, data_buffer in enumerate(data):
            data_input_dict = prepare_batch_input(
                data_buffer, data_input_names, ModelHyperParams.bos_idx,
                ModelHyperParams.bos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)
            feed_dict_list.append(data_input_dict)
    return feed_dict_list if len(feed_dict_list) == count else None


def prepare_dense_feed_dict_list(data_generator, count):
    """
    Prepare the list of feed dict for multi-devices.
    """
    feed_dict_list = []
    if data_generator is not None:  # use_py_reader == False
        data_input_names = dense_encoder_data_input_fields + fast_decoder_data_input_fields
        data = next(data_generator)
        for idx, data_buffer in enumerate(data):
            data_input_dict = prepare_batch_input(
                data_buffer, data_input_names, DenseModelHyperParams.bos_idx,
                DenseModelHyperParams.bos_idx, DenseModelHyperParams.n_head,
                DenseModelHyperParams.d_model)
            feed_dict_list.append(data_input_dict)
    return feed_dict_list if len(feed_dict_list) == count else None


def prepare_infer_feed_dict_list(data_generator, count):
    feed_dict_list = []
    if data_generator is not None:  # use_py_reader == False
        data_input_names = encoder_data_input_fields + fast_decoder_data_input_fields
        dense_data_input_names = dense_encoder_data_input_fields + fast_decoder_data_input_fields
        data = next(data_generator)
        for idx, data_buffer in enumerate(data):
            dense_data_input_dict = prepare_batch_input(
                data_buffer, dense_data_input_names, DenseModelHyperParams.bos_idx,
                DenseModelHyperParams.bos_idx, DenseModelHyperParams.n_head,
                DenseModelHyperParams.d_model)

            data_input_dict = prepare_batch_input(data_buffer, data_input_names, 
                ModelHyperParams.bos_idx, ModelHyperParams.bos_idx, 
                ModelHyperParams.n_head, ModelHyperParams.d_model)
            
            for key in dense_data_input_dict:
                if key not in data_input_dict:
                    data_input_dict[key] = dense_data_input_dict[key]
            
            feed_dict_list.append(data_input_dict)
    return feed_dict_list if len(feed_dict_list) == count else None



def get_trans_res(batch_size, out_list, final_list):
    """
        Get trans
    """
    for index in xrange(batch_size):
        seq = out_list[index][0] #top1 seq

        if 1 not in seq:
            res = seq[1:-1]
        else:
            res = seq[1:seq.index(1)]

        res = map(str, res)
        final_list.append(" ".join(res))


def fast_infer(args):
    """
    Inference by beam search decoder based solely on Fluid operators.
    """
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    #with fluid.program_guard(test_prog, startup_prog):
    with fluid.unique_name.guard("new_forward"):
        out_ids1, out_scores1 = forward_fast_decode(
            ModelHyperParams.src_vocab_size,
            ModelHyperParams.trg_vocab_size,
            ModelHyperParams.max_length + 50,
            ModelHyperParams.n_layer,
            ModelHyperParams.n_head,
            ModelHyperParams.d_key,
            ModelHyperParams.d_value,
            ModelHyperParams.d_model,
            ModelHyperParams.d_inner_hid,
            ModelHyperParams.prepostprocess_dropout,
            ModelHyperParams.attention_dropout,
            ModelHyperParams.relu_dropout,
            ModelHyperParams.preprocess_cmd,
            ModelHyperParams.postprocess_cmd,
            ModelHyperParams.weight_sharing,
            ModelHyperParams.embedding_sharing,
            InferTaskConfig.beam_size,
            args.infer_batch_size,
            InferTaskConfig.max_out_len,
            args.decode_alpha,
            ModelHyperParams.eos_idx,
            params_type="new"
            )
    
    with fluid.unique_name.guard("new_relative_position"):
        out_ids2, out_scores2 = relative_fast_decode(
            ModelHyperParams.src_vocab_size,
            ModelHyperParams.trg_vocab_size,
            ModelHyperParams.max_length + 50,
            ModelHyperParams.n_layer,
            ModelHyperParams.n_head,
            ModelHyperParams.d_key,
            ModelHyperParams.d_value,
            ModelHyperParams.d_model,
            ModelHyperParams.d_inner_hid,
            ModelHyperParams.prepostprocess_dropout,
            ModelHyperParams.attention_dropout,
            ModelHyperParams.relu_dropout,
            ModelHyperParams.preprocess_cmd,
            ModelHyperParams.postprocess_cmd,
            ModelHyperParams.weight_sharing,
            ModelHyperParams.embedding_sharing,
            InferTaskConfig.beam_size,
            args.infer_batch_size,
            InferTaskConfig.max_out_len,
            args.decode_alpha,
            ModelHyperParams.eos_idx,
            params_type="new"
            )

    DenseModelHyperParams.src_vocab_size = ModelHyperParams.src_vocab_size
    DenseModelHyperParams.trg_vocab_size = ModelHyperParams.trg_vocab_size
    DenseModelHyperParams.weight_sharing = ModelHyperParams.weight_sharing
    DenseModelHyperParams.embedding_sharing = ModelHyperParams.embedding_sharing 
    with fluid.unique_name.guard("new_dense"):
        out_ids3, out_scores3 = dense_fast_decode(
            DenseModelHyperParams.src_vocab_size,
            DenseModelHyperParams.trg_vocab_size,
            DenseModelHyperParams.max_length + 50,
            DenseModelHyperParams.n_layer,
            DenseModelHyperParams.enc_n_layer,
            DenseModelHyperParams.n_head,
            DenseModelHyperParams.d_key,
            DenseModelHyperParams.d_value,
            DenseModelHyperParams.d_model,
            DenseModelHyperParams.d_inner_hid,
            DenseModelHyperParams.prepostprocess_dropout,
            DenseModelHyperParams.attention_dropout,
            DenseModelHyperParams.relu_dropout,
            DenseModelHyperParams.preprocess_cmd,
            DenseModelHyperParams.postprocess_cmd,
            DenseModelHyperParams.weight_sharing,
            DenseModelHyperParams.embedding_sharing,
            InferTaskConfig.beam_size,
            args.infer_batch_size,
            InferTaskConfig.max_out_len,
            args.decode_alpha,
            ModelHyperParams.eos_idx,
            params_type="new"
            )

    test_prog = fluid.default_main_program().clone(for_test=True)
    # This is used here to set dropout to the test mode.

    if InferTaskConfig.use_gpu:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.load_params(
        exe,
        InferTaskConfig.model_path,
        main_program=test_prog)


    if args.use_mem_opt:
        fluid.memory_optimize(test_prog)

    exec_strategy = fluid.ExecutionStrategy()
    # For faster executor
    exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 1
    build_strategy = fluid.BuildStrategy()
    
    # data reader settings for inference
    args.use_token_batch = False
    #args.sort_type = reader.SortType.NONE
    args.shuffle = False
    args.shuffle_batch = False
    
    dev_count = 1
    lines_cnt = len(open(args.val_file_pattern, 'r').readlines())
    data_reader = line_reader(args.val_file_pattern, args.infer_batch_size, dev_count,
                    token_delimiter=args.token_delimiter,
                    max_len=ModelHyperParams.max_length,
                    parse_line=parse_src_line)

    test_data = prepare_data_generator(
        args,
        is_test=True,
        count=dev_count,
        pyreader=None,
        batch_size=args.infer_batch_size, data_reader=data_reader)
    
    data_generator = test_data()
    iter_num = 0

    if not os.path.exists("trans"):
        os.mkdir("trans")
    
    model_name = InferTaskConfig.model_path.split("/")[-1]
    forward_res = open(os.path.join("trans", "forward_%s" % model_name), 'w')
    relative_res = open(os.path.join("trans", "relative_%s" % model_name), 'w')
    dense_res = open(os.path.join("trans", "dense_%s" % model_name), 'w')

    forward_list = []
    relative_list = []
    dense_list = []
    with profile_context(False):
        while True:
            try:
                feed_dict_list = prepare_infer_feed_dict_list(data_generator, dev_count)

                forward_seq_ids, relative_seq_ids, dense_seq_ids = exe.run(
                    program=test_prog,
                    fetch_list=[out_ids1.name, out_ids2.name, out_ids3.name],
                    feed=feed_dict_list[0]
                    if feed_dict_list is not None else None,
                    return_numpy=False,
                    use_program_cache=False)

                fseq_ids = np.asarray(forward_seq_ids).tolist()
                rseq_ids = np.asarray(relative_seq_ids).tolist()
                dseq_ids = np.asarray(dense_seq_ids).tolist()
                
                get_trans_res(args.infer_batch_size, fseq_ids, forward_list)
                get_trans_res(args.infer_batch_size, rseq_ids, relative_list)
                get_trans_res(args.infer_batch_size, dseq_ids, dense_list)

                
            except (StopIteration, fluid.core.EOFException):
                break
        forward_list = forward_list[:lines_cnt]
        relative_list = relative_list[:lines_cnt]
        dense_list = dense_list[:lines_cnt]

        forward_res.writelines("\n".join(forward_list))
        forward_res.flush()
        forward_res.close()

        relative_res.writelines("\n".join(relative_list))
        relative_res.flush()
        relative_res.close()

        dense_res.writelines("\n".join(dense_list))
        dense_res.flush()
        dense_res.close()


@contextlib.contextmanager
def profile_context(profile=True):
    """
        profile_context
    """
    if profile:
        with profiler.profiler('All', 'total', './profile_dir/profile_file_tmp'):
            yield
    else:
        yield


if __name__ == "__main__":
    args = parse_args()
    fast_infer(args)

