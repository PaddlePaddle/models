import os
import time
import sys

import argparse
import logging
import numpy as np
import yaml
from attrdict import AttrDict
from pprint import pprint

import paddle

from paddlenlp.transformers import InferTransformerModel

sys.path.append("../")
import reader

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../configs/transformer.big.yaml",
        type=str,
        help="Path of the config file. ")
    args = parser.parse_args()
    return args


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


def do_train(args):
    paddle.enable_static()
    if args.use_gpu:
        place = paddle.set_device("gpu:0")
    else:
        place = paddle.set_device("cpu")

    # Define data loader
    test_loader, to_tokens = reader.create_infer_loader(args)

    test_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(test_program, startup_program):
        src_word = paddle.static.data(
            name="src_word", shape=[None, None], dtype="int64")

        # Define model
        transformer = InferTransformerModel(
            src_vocab_size=args.src_vocab_size,
            trg_vocab_size=args.trg_vocab_size,
            max_length=args.max_length + 1,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_model=args.d_model,
            d_inner_hid=args.d_inner_hid,
            dropout=args.dropout,
            weight_sharing=args.weight_sharing,
            bos_id=args.bos_idx,
            eos_id=args.eos_idx,
            beam_size=args.beam_size,
            max_out_len=args.max_out_len)

        finished_seq = transformer(src_word=src_word)

    test_program = test_program.clone(for_test=True)

    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    amp_list = paddle.static.amp.AutoMixedPrecisionLists(
                custom_white_list=['softmax', 'layer_norm'],
                custom_black_list=['tril_triu'])
    paddle.static.amp.cast_model_to_fp16(test_program, amp_list)

    assert (
        args.init_from_params), "must set init_from_params to load parameters"
    paddle.static.load(test_program,
                       os.path.join(args.init_from_params, "transformer"), exe)
    print("finish initing model from params from %s" % (args.init_from_params))

    f = open(args.output_file, "w")
    for data in test_loader:
        finished_sequence, = exe.run(test_program,
                                     feed={'src_word': data[0]},
                                     fetch_list=finished_seq.name)
        finished_sequence = finished_sequence.transpose([0, 2, 1])
        for ins in finished_sequence:
            for beam_idx, beam in enumerate(ins):
                if beam_idx >= args.n_best:
                    break
                id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                word_list = to_tokens(id_list)
                sequence = " ".join(word_list) + "\n"
                f.write(sequence)

    paddle.disable_static()


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_train(args)
