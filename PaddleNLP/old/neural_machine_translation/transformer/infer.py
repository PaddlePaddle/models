import argparse
import ast
import multiprocessing
import numpy as np
import os
from functools import partial

import paddle
import paddle.fluid as fluid

import model
import reader
from config import *
from model import wrap_encoder as encoder
from model import wrap_decoder as decoder
from model import fast_decode as fast_decoder
from train import pad_batch_data, prepare_data_generator


def parse_args():
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        "--src_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of source language.")
    parser.add_argument(
        "--trg_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of target language.")
    parser.add_argument(
        "--test_file_pattern",
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
        default=True,
        help="The flag indicating whether to use py_reader.")
    parser.add_argument(
        "--use_parallel_exe",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use ParallelExecutor.")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Append args related to dict
    src_dict = reader.DataReader.load_dict(args.src_vocab_fpath)
    trg_dict = reader.DataReader.load_dict(args.trg_vocab_fpath)
    dict_args = [
        "src_vocab_size", str(len(src_dict)), "trg_vocab_size",
        str(len(trg_dict)), "bos_idx", str(src_dict[args.special_token[0]]),
        "eos_idx", str(src_dict[args.special_token[1]]), "unk_idx",
        str(src_dict[args.special_token[2]])
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
                        d_model, place):
    """
    Put all padded data needed by beam search decoder into a dict.
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

    def to_lodtensor(data, place, lod=None):
        data_tensor = fluid.LoDTensor()
        data_tensor.set(data, place)
        if lod is not None:
            data_tensor.set_lod(lod)
        return data_tensor

    # beamsearch_op must use tensors with lod
    init_score = to_lodtensor(
        np.zeros_like(
            trg_word, dtype="float32").reshape(-1, 1),
        place, [range(trg_word.shape[0] + 1)] * 2)
    trg_word = to_lodtensor(trg_word, place, [range(trg_word.shape[0] + 1)] * 2)
    init_idx = np.asarray(range(len(insts)), dtype="int32")

    data_input_dict = dict(
        zip(data_input_names, [
            src_word, src_pos, src_slf_attn_bias, trg_word, init_score,
            init_idx, trg_src_attn_bias
        ]))
    return data_input_dict


def prepare_feed_dict_list(data_generator, count, place):
    """
    Prepare the list of feed dict for multi-devices.
    """
    feed_dict_list = []
    if data_generator is not None:  # use_py_reader == False
        data_input_names = encoder_data_input_fields + fast_decoder_data_input_fields
        data = next(data_generator)
        for idx, data_buffer in enumerate(data):
            data_input_dict = prepare_batch_input(
                data_buffer, data_input_names, ModelHyperParams.eos_idx,
                ModelHyperParams.bos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model, place)
            feed_dict_list.append(data_input_dict)
    return feed_dict_list if len(feed_dict_list) == count else None


def py_reader_provider_wrapper(data_reader, place):
    """
    Data provider needed by fluid.layers.py_reader.
    """

    def py_reader_provider():
        data_input_names = encoder_data_input_fields + fast_decoder_data_input_fields
        for batch_id, data in enumerate(data_reader()):
            data_input_dict = prepare_batch_input(
                data, data_input_names, ModelHyperParams.eos_idx,
                ModelHyperParams.bos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model, place)
            yield [data_input_dict[item] for item in data_input_names]

    return py_reader_provider


def fast_infer(args):
    """
    Inference by beam search decoder based solely on Fluid operators.
    """
    out_ids, out_scores, pyreader = fast_decoder(
        ModelHyperParams.src_vocab_size,
        ModelHyperParams.trg_vocab_size,
        ModelHyperParams.max_length + 1,
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
        InferTaskConfig.beam_size,
        InferTaskConfig.max_out_len,
        ModelHyperParams.eos_idx,
        use_py_reader=args.use_py_reader)

    # This is used here to set dropout to the test mode.
    infer_program = fluid.default_main_program().clone(for_test=True)

    if args.use_mem_opt:
        fluid.memory_optimize(infer_program)

    if InferTaskConfig.use_gpu:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.load_vars(
        exe,
        InferTaskConfig.model_path,
        vars=[
            var for var in infer_program.list_vars()
            if isinstance(var, fluid.framework.Parameter)
        ])

    exec_strategy = fluid.ExecutionStrategy()
    # For faster executor
    exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 1
    build_strategy = fluid.BuildStrategy()
    infer_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        main_program=infer_program,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    # data reader settings for inference
    args.train_file_pattern = args.test_file_pattern
    args.use_token_batch = False
    args.sort_type = reader.SortType.NONE
    args.shuffle = False
    args.shuffle_batch = False
    test_data = prepare_data_generator(
        args,
        is_test=False,
        count=dev_count,
        pyreader=pyreader,
        py_reader_provider_wrapper=py_reader_provider_wrapper,
        place=place)
    if args.use_py_reader:
        pyreader.start()
        data_generator = None
    else:
        data_generator = test_data()
    trg_idx2word = reader.DataReader.load_dict(
        dict_path=args.trg_vocab_fpath, reverse=True)

    while True:
        try:
            feed_dict_list = prepare_feed_dict_list(data_generator, dev_count,
                                                    place)
            if args.use_parallel_exe:
                seq_ids, seq_scores = infer_exe.run(
                    fetch_list=[out_ids.name, out_scores.name],
                    feed=feed_dict_list,
                    return_numpy=False)
            else:
                seq_ids, seq_scores = exe.run(
                    program=infer_program,
                    fetch_list=[out_ids.name, out_scores.name],
                    feed=feed_dict_list[0]
                    if feed_dict_list is not None else None,
                    return_numpy=False,
                    use_program_cache=True)
            seq_ids_list, seq_scores_list = [
                seq_ids
            ], [seq_scores] if isinstance(
                seq_ids, paddle.fluid.LoDTensor) else (seq_ids, seq_scores)
            for seq_ids, seq_scores in zip(seq_ids_list, seq_scores_list):
                # How to parse the results:
                #   Suppose the lod of seq_ids is:
                #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
                #   then from lod[0]:
                #     there are 2 source sentences, beam width is 3.
                #   from lod[1]:
                #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
                #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
                hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
                scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
                for i in range(len(seq_ids.lod()[0]) -
                               1):  # for each source sentence
                    start = seq_ids.lod()[0][i]
                    end = seq_ids.lod()[0][i + 1]
                    for j in range(end - start):  # for each candidate
                        sub_start = seq_ids.lod()[1][start + j]
                        sub_end = seq_ids.lod()[1][start + j + 1]
                        hyps[i].append(" ".join([
                            trg_idx2word[idx]
                            for idx in post_process_seq(
                                np.array(seq_ids)[sub_start:sub_end])
                        ]))
                        scores[i].append(np.array(seq_scores)[sub_end - 1])
                        print(hyps[i][-1])
                        if len(hyps[i]) >= InferTaskConfig.n_best:
                            break
        except (StopIteration, fluid.core.EOFException):
            # The data pass is over.
            if args.use_py_reader:
                pyreader.reset()
            break


if __name__ == "__main__":
    args = parse_args()
    fast_infer(args)
