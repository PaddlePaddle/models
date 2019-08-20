#encoding=utf8
import logging
import os
import six
import sys
import time

import numpy as np
import paddle
import paddle.fluid as fluid

#include palm for easier nlp coding
from palm.toolkit.input_field import InputField
from palm.toolkit.configure import PDConfig

# include task-specific libs
import desc
import reader
from transformer import create_net, position_encoding_init


def init_from_pretrain_model(args, exe, program):

    assert isinstance(args.init_from_pretrain_model, str)

    if not os.path.exists(args.init_from_pretrain_model):
        raise Warning("The pretrained params do not exist.")
        return False

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(
            os.path.join(args.init_from_pretrain_model, var.name))

    fluid.io.load_vars(
        exe,
        args.init_from_pretrain_model,
        main_program=program,
        predicate=existed_params)

    print("finish initing model from pretrained params from %s" %
          (args.init_from_pretrain_model))

    return True


def init_from_params(args, exe, program):

    assert isinstance(args.init_from_params, str)

    if not os.path.exists(args.init_from_params):
        raise Warning("the params path does not exist.")
        return False

    fluid.io.load_params(
        executor=exe,
        dirname=args.init_from_params,
        main_program=program,
        filename="params.pdparams")

    print("finish init model from params from %s" % (args.init_from_params))

    return True


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
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


def do_predict(args):
    if args.use_cuda:
        dev_count = fluid.core.get_cuda_device_count()
        place = fluid.CUDAPlace(0)
    else:
        dev_count = int(os.environ.get('CPU_NUM', 1))
        place = fluid.CPUPlace()
    # define the data generator
    processor = reader.DataProcessor(
        fpattern=args.predict_file,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=False,
        batch_size=args.batch_size,
        device_count=dev_count,
        pool_size=args.pool_size,
        sort_type=reader.SortType.NONE,
        shuffle=False,
        shuffle_batch=False,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        max_length=args.max_length,
        n_head=args.n_head)
    batch_generator = processor.data_generator(phase="predict", place=place)
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = processor.get_vocab_summary()
    trg_idx2word = reader.DataProcessor.load_dict(
        dict_path=args.trg_vocab_fpath, reverse=True)

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():

            # define input and reader

            input_field_names = desc.encoder_data_input_fields + desc.fast_decoder_data_input_fields
            input_slots = [{
                "name": name,
                "shape": desc.input_descs[name][0],
                "dtype": desc.input_descs[name][1]
            } for name in input_field_names]

            input_field = InputField(input_slots)
            input_field.build(build_pyreader=True)

            # define the network

            predictions = create_net(
                is_training=False, model_input=input_field, args=args)
            out_ids, out_scores = predictions

            out_ids.persistable = out_scores.persistable = True
    # This is used here to set dropout to the test mode.
    test_prog = test_prog.clone(for_test=True)

    # prepare predicting

    ## define the executor and program for training

    exe = fluid.Executor(place)

    exe.run(startup_prog)
    assert (args.init_from_params) or (args.init_from_pretrain_model)

    if args.init_from_params:
        init_from_params(args, exe, test_prog)

    elif args.init_from_pretrain_model:
        init_from_pretrain_model(args, exe, test_prog)

    # to avoid a longer length than training, reset the size of position encoding to max_length
    for pos_enc_param_name in desc.pos_enc_param_names:
        pos_enc_param = fluid.global_scope().find_var(
            pos_enc_param_name).get_tensor()

        pos_enc_param.set(
            position_encoding_init(args.max_length + 1, args.d_model), place)

    compiled_test_prog = fluid.CompiledProgram(test_prog)

    f = open(args.output_file, "wb")
    # start predicting
    ## decorate the pyreader with batch_generator
    input_field.reader.decorate_batch_generator(batch_generator)
    input_field.reader.start()
    while True:
        try:
            seq_ids, seq_scores = exe.run(
                compiled_test_prog,
                fetch_list=[out_ids.name, out_scores.name],
                return_numpy=False)

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
                    hyps[i].append(b" ".join([
                        trg_idx2word[idx]
                        for idx in post_process_seq(
                            np.array(seq_ids)[sub_start:sub_end], args.bos_idx,
                            args.eos_idx)
                    ]))
                    scores[i].append(np.array(seq_scores)[sub_end - 1])
                    f.write(hyps[i][-1] + b"\n")
                    if len(hyps[i]) >= args.n_best:
                        break
        except fluid.core.EOFException:
            break

    f.close()


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()

    do_predict(args)
