import argparse
import ast
import multiprocessing
import os
import six
import time

import numpy as np
import paddle.fluid as fluid

import reader
from config import *
from model import transformer, position_encoding_init


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
        "--train_file_pattern",
        type=str,
        required=True,
        help="The pattern to match training data files.")
    parser.add_argument(
        "--val_file_pattern",
        type=str,
        help="The pattern to match validation data files.")
    parser.add_argument(
        "--use_token_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to "
        "produce batch data according to token number.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="The number of sequences contained in a mini-batch, or the maximum "
        "number of tokens (include paddings) contained in a mini-batch. Note "
        "that this represents the number on single device and the actual batch "
        "size for multi-devices will multiply the device number.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=200000,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--sort_type",
        default="pool",
        choices=("global", "pool", "none"),
        help="The grain to sort by length: global for all instances; pool for "
        "instances in pool; none for no sort.")
    parser.add_argument(
        "--shuffle",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument(
        "--shuffle_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle the data batches.")
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
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--local',
        type=ast.literal_eval,
        default=True,
        help='Whether to run as local mode.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help="The device type.")
    parser.add_argument(
        '--sync', type=ast.literal_eval, default=True, help="sync mode.")
    parser.add_argument(
        "--enable_ce",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to run the task "
        "for continuous evaluation.")
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
                        [TrainTaskConfig, ModelHyperParams])
    return args


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array(
            [[1.] * len(inst) + [0.] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array([
            list(range(0, len(inst))) + [0] * (max_len - len(inst))
            for inst in insts
        ])
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_input(insts, data_input_names, src_pad_idx, trg_pad_idx,
                        n_head, d_model):
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len, 1)
    trg_pos = trg_pos.reshape(-1, trg_max_len, 1)

    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)

    data_input_dict = dict(
        zip(data_input_names, [
            src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
            trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
        ]))

    return data_input_dict, np.asarray([num_token], dtype="float32")


def prepare_data_generator(args, is_test, count, pyreader):
    """
    Data generator wrapper for DataReader. If use py_reader, set the data
    provider for py_reader
    """
    data_reader = reader.DataReader(
        fpattern=args.val_file_pattern if is_test else args.train_file_pattern,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size * (1 if args.use_token_batch else count),
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        shuffle=args.shuffle,
        shuffle_batch=args.shuffle_batch,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False).batch_generator

    def stack(data_reader, count, clip_last=True):
        def __impl__():
            res = []
            for item in data_reader():
                res.append(item)
                if len(res) == count:
                    yield res
                    res = []
            if len(res) == count:
                yield res
            elif not clip_last:
                data = []
                for item in res:
                    data += item
                if len(data) > count:
                    inst_num_per_part = len(data) // count
                    yield [
                        data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
                        for i in range(count)
                    ]

        return __impl__

    def split(data_reader, count):
        def __impl__():
            for item in data_reader():
                inst_num_per_part = len(item) // count
                for i in range(count):
                    yield item[inst_num_per_part * i:inst_num_per_part * (i + 1
                                                                          )]

        return __impl__

    if not args.use_token_batch:
        # to make data on each device have similar token number
        data_reader = split(data_reader, count)
    if args.use_py_reader:
        pyreader.decorate_tensor_provider(
            py_reader_provider_wrapper(data_reader))
        data_reader = None
    else:  # Data generator for multi-devices
        data_reader = stack(data_reader, count)
    return data_reader


def prepare_feed_dict_list(data_generator, init_flag, count):
    """
    Prepare the list of feed dict for multi-devices.
    """
    feed_dict_list = []
    if data_generator is not None:  # use_py_reader == False
        data_input_names = encoder_data_input_fields + \
                    decoder_data_input_fields[:-1] + label_data_input_fields
        data = next(data_generator)
        for idx, data_buffer in enumerate(data):
            data_input_dict, num_token = prepare_batch_input(
                data_buffer, data_input_names, ModelHyperParams.eos_idx,
                ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)
            feed_dict_list.append(data_input_dict)
    if init_flag:
        for idx in range(count):
            pos_enc_tables = dict()
            for pos_enc_param_name in pos_enc_param_names:
                pos_enc_tables[pos_enc_param_name] = position_encoding_init(
                    ModelHyperParams.max_length + 1, ModelHyperParams.d_model)
            if len(feed_dict_list) <= idx:
                feed_dict_list.append(pos_enc_tables)
            else:
                feed_dict_list[idx] = dict(
                    list(pos_enc_tables.items()) + list(feed_dict_list[idx]
                                                        .items()))

    return feed_dict_list if len(feed_dict_list) == count else None


def py_reader_provider_wrapper(data_reader):
    """
    Data provider needed by fluid.layers.py_reader.
    """

    def py_reader_provider():
        data_input_names = encoder_data_input_fields + \
                    decoder_data_input_fields[:-1] + label_data_input_fields
        for batch_id, data in enumerate(data_reader()):
            data_input_dict, num_token = prepare_batch_input(
                data, data_input_names, ModelHyperParams.eos_idx,
                ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)
            total_dict = dict(data_input_dict.items())
            yield [total_dict[item] for item in data_input_names]

    return py_reader_provider


def test_context(exe, train_exe, dev_count):
    # Context to do validation.
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    if args.enable_ce:
        test_prog.random_seed = 1000
        startup_prog.random_seed = 1000
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            sum_cost, avg_cost, predict, token_num, pyreader = transformer(
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
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=True)

    test_data = prepare_data_generator(
        args, is_test=True, count=dev_count, pyreader=pyreader)

    exe.run(startup_prog)
    test_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        main_program=test_prog,
        share_vars_from=train_exe)

    def test(exe=test_exe, pyreader=pyreader):
        test_total_cost = 0
        test_total_token = 0

        if args.use_py_reader:
            pyreader.start()
            data_generator = None
        else:
            data_generator = test_data()
        while True:
            try:
                feed_dict_list = prepare_feed_dict_list(data_generator, False,
                                                        dev_count)
                outs = test_exe.run(fetch_list=[sum_cost.name, token_num.name],
                                    feed=feed_dict_list)
            except (StopIteration, fluid.core.EOFException):
                # The current pass is over.
                if args.use_py_reader:
                    pyreader.reset()
                break
            sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
            test_total_cost += sum_cost_val.sum()
            test_total_token += token_num_val.sum()
        test_avg_cost = test_total_cost / test_total_token
        test_ppl = np.exp([min(test_avg_cost, 100)])
        return test_avg_cost, test_ppl

    return test


def train_loop(exe, train_prog, startup_prog, dev_count, sum_cost, avg_cost,
               token_num, predict, pyreader):
    # Initialize the parameters.
    if TrainTaskConfig.ckpt_path:
        fluid.io.load_persistables(exe, TrainTaskConfig.ckpt_path)
    else:
        print("init fluid.framework.default_startup_program")
        exe.run(startup_prog)

    train_data = prepare_data_generator(
        args, is_test=False, count=dev_count, pyreader=pyreader)

    # For faster executor
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.use_experimental_executor = True
    # exec_strategy.num_iteration_per_drop_scope = 5
    build_strategy = fluid.BuildStrategy()
    # Since the token number differs among devices, customize gradient scale to
    # use token average cost among multi-devices. and the gradient scale is
    # `1 / token_number` for average cost.
    # build_strategy.gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized
    train_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        loss_name=avg_cost.name,
        main_program=train_prog,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    if args.val_file_pattern is not None:
        test = test_context(exe, train_exe, dev_count)

    # the best cross-entropy value with label smoothing
    loss_normalizer = -((1. - TrainTaskConfig.label_smooth_eps) * np.log(
        (1. - TrainTaskConfig.label_smooth_eps
         )) + TrainTaskConfig.label_smooth_eps *
                        np.log(TrainTaskConfig.label_smooth_eps / (
                            ModelHyperParams.trg_vocab_size - 1) + 1e-20))

    step_idx = 0
    init_flag = True
    for pass_id in six.moves.xrange(TrainTaskConfig.pass_num):
        pass_start_time = time.time()

        if args.use_py_reader:
            pyreader.start()
            data_generator = None
        else:
            data_generator = train_data()

        batch_id = 0
        while True:
            try:
                feed_dict_list = prepare_feed_dict_list(data_generator,
                                                        init_flag, dev_count)

                outs = train_exe.run(
                    fetch_list=[sum_cost.name, token_num.name],
                    feed=feed_dict_list)
                sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[
                    1])
                # sum the cost from multi-devices
                total_sum_cost = sum_cost_val.sum()
                total_token_num = token_num_val.sum()
                total_avg_cost = total_sum_cost / total_token_num

                print("step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                      "normalized loss: %f, ppl: %f" %
                      (step_idx, pass_id, batch_id, total_avg_cost,
                       total_avg_cost - loss_normalizer,
                       np.exp([min(total_avg_cost, 100)])))

                if step_idx % int(TrainTaskConfig.
                                  save_freq) == TrainTaskConfig.save_freq - 1:
                    fluid.io.save_persistables(
                        exe,
                        os.path.join(TrainTaskConfig.ckpt_dir,
                                     "latest.checkpoint"), train_prog)
                    fluid.io.save_params(
                        exe,
                        os.path.join(TrainTaskConfig.model_dir,
                                     "iter_" + str(step_idx) + ".infer.model"),
                        train_prog)
                init_flag = False
                batch_id += 1
                step_idx += 1
            except (StopIteration, fluid.core.EOFException):
                # The current pass is over.
                if args.use_py_reader:
                    pyreader.reset()
                break

        time_consumed = time.time() - pass_start_time
        # Validate and save the persistable.
        if args.val_file_pattern is not None:
            val_avg_cost, val_ppl = test()
            print(
                "epoch: %d, val avg loss: %f, val normalized loss: %f, val ppl: %f,"
                " consumed %fs" % (pass_id, val_avg_cost,
                                   val_avg_cost - loss_normalizer, val_ppl,
                                   time_consumed))
        else:
            print("epoch: %d, consumed %fs" % (pass_id, time_consumed))
        if not args.enable_ce:
            fluid.io.save_persistables(
                exe,
                os.path.join(TrainTaskConfig.ckpt_dir,
                             "pass_" + str(pass_id) + ".checkpoint"),
                train_prog)

    if args.enable_ce:  # For CE
        print("kpis\ttrain_cost_card%d\t%f" % (dev_count, total_avg_cost))
        if args.val_file_pattern is not None:
            print("kpis\ttest_cost_card%d\t%f" % (dev_count, val_avg_cost))
        print("kpis\ttrain_duration_card%d\t%f" % (dev_count, time_consumed))


def train(args):
    # priority: ENV > args > config
    is_local = os.getenv("PADDLE_IS_LOCAL", "1")
    if is_local == '0':
        args.local = False
    print(args)

    if args.device == 'CPU':
        TrainTaskConfig.use_gpu = False

    training_role = os.getenv("TRAINING_ROLE", "TRAINER")

    if training_role == "PSERVER" or (not TrainTaskConfig.use_gpu):
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()

    exe = fluid.Executor(place)

    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    if args.enable_ce:
        train_prog.random_seed = 1000
        startup_prog.random_seed = 1000

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            sum_cost, avg_cost, predict, token_num, pyreader = transformer(
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
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=False)

            if args.local:
                lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
                    ModelHyperParams.d_model, TrainTaskConfig.warmup_steps)
                optimizer = fluid.optimizer.Adam(
                    learning_rate=lr_decay * TrainTaskConfig.learning_rate,
                    beta1=TrainTaskConfig.beta1,
                    beta2=TrainTaskConfig.beta2,
                    epsilon=TrainTaskConfig.eps)
            elif args.sync == False:
                optimizer = fluid.optimizer.SGD(0.003)
            optimizer.minimize(avg_cost)

    if args.use_mem_opt:
        fluid.memory_optimize(train_prog)

    if args.local:
        print("local start_up:")
        train_loop(exe, train_prog, startup_prog, dev_count, sum_cost, avg_cost,
                   token_num, predict, pyreader)
    else:
        port = os.getenv("PADDLE_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers,
            program=train_prog,
            startup_program=startup_prog)

        if training_role == "PSERVER":
            current_endpoint = os.getenv("POD_IP") + ":" + os.getenv(
                "PADDLE_PORT")
            if not current_endpoint:
                print("need env SERVER_ENDPOINT")
                exit(1)
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)

            print("psserver begin run")
            with open('pserver_startup.desc', 'w') as f:
                f.write(str(pserver_startup))
            with open('pserver_prog.desc', 'w') as f:
                f.write(str(pserver_prog))
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            trainer_prog = t.get_trainer_program()
            with open('trainer_prog.desc', 'w') as f:
                f.write(str(trainer_prog))
            train_loop(exe, train_prog, startup_prog, dev_count, sum_cost,
                       avg_cost, token_num, predict, pyreader)
        else:
            print("environment var TRAINER_ROLE should be TRAINER os PSERVER")


if __name__ == "__main__":
    args = parse_args()
    train(args)
