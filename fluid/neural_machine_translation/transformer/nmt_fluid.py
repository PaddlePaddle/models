import os
import numpy as np
import time
import argparse

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

from model import transformer, position_encoding_init
from optim import LearningRateScheduler
from config import TrainTaskConfig, ModelHyperParams, pos_enc_param_names, \
        encoder_input_data_names, decoder_input_data_names, label_data_names
import paddle.fluid.debuger as debuger

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--batch_size', type=int, default=TrainTaskConfig.batch_size, help="Batch size for training.")

parser.add_argument(
    '--learning_rate',
    type=float,
    default=TrainTaskConfig.learning_rate,
    help="Learning rate for training.")

parser.add_argument('--num_passes', type=int, default=50, help="No. of passes.")

parser.add_argument(
    '--device',
    type=str,
    default='CPU',
    choices=['CPU', 'GPU'],
    help="The device type.")

parser.add_argument('--device_id', type=int, default=0, help="The device id.")

parser.add_argument(
    '--local',
    type=str2bool,
    default=True,
    help='Whether to run as local mode.')

parser.add_argument(
    "--ps_hosts",
    type=str,
    default="",
    help="Comma-separated list of hostname:port pairs")

parser.add_argument(
    "--trainer_hosts",
    type=str,
    default="",
    help="Comma-separated list of hostname:port pairs")

parser.add_argument(
    "--pass_num",
    type=int,
    default=TrainTaskConfig.pass_num,
    help="pass num of train")

# Flags for defining the tf.train.Server
parser.add_argument(
    "--task_index", type=int, default=0, help="Index of task within the job")
args = parser.parse_args()

def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   return_pos=True,
                   return_attn_bias=True,
                   return_max_len=True):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if return_pos:
        inst_pos = np.array([[
            pos_i + 1 if w_i != pad_idx else 0 for pos_i, w_i in enumerate(inst)
        ] for inst in inst_data])

        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                [-1, 1, max_len, max_len])
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
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_input(insts, input_data_names, src_pad_idx, trg_pad_idx,
                        max_length, n_head):
    print("input_data_name:", input_data_names)
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")
    src_slf_attn_pre_softmax_shape = np.array(
        [-1, src_slf_attn_bias.shape[-1]], dtype="int32")
    src_slf_attn_post_softmax_shape = np.array(
        src_slf_attn_bias.shape, dtype="int32")
    trg_slf_attn_pre_softmax_shape = np.array(
        [-1, trg_slf_attn_bias.shape[-1]], dtype="int32")
    trg_slf_attn_post_softmax_shape = np.array(
        trg_slf_attn_bias.shape, dtype="int32")
    trg_src_attn_pre_softmax_shape = np.array(
        [-1, trg_src_attn_bias.shape[-1]], dtype="int32")
    trg_src_attn_post_softmax_shape = np.array(
        trg_src_attn_bias.shape, dtype="int32")
    lbl_word = pad_batch_data([inst[2] for inst in insts], trg_pad_idx, n_head,
                              False, False, False, False)
    lbl_weight = (lbl_word != trg_pad_idx).astype("float32").reshape([-1, 1])
    input_dict = dict(
        zip(input_data_names, [
            src_word, src_pos, src_slf_attn_bias,
            src_slf_attn_pre_softmax_shape, src_slf_attn_post_softmax_shape,
            trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias,
            trg_slf_attn_pre_softmax_shape, trg_slf_attn_post_softmax_shape,
            trg_src_attn_pre_softmax_shape, trg_src_attn_post_softmax_shape,
            lbl_word, lbl_weight
        ]))
    #print("input_dict", input_dict)
    return input_dict


def main():
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(
        args.device_id)
    exe = fluid.Executor(place)

    cost, predict = transformer(
        ModelHyperParams.src_vocab_size + 1,
        ModelHyperParams.trg_vocab_size + 1, ModelHyperParams.max_length + 1,
        ModelHyperParams.n_layer, ModelHyperParams.n_head,
        ModelHyperParams.d_key, ModelHyperParams.d_value,
        ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
        ModelHyperParams.dropout, ModelHyperParams.src_pad_idx,
        ModelHyperParams.trg_pad_idx, ModelHyperParams.pos_pad_idx)

    lr_scheduler = LearningRateScheduler(ModelHyperParams.d_model,
                                         TrainTaskConfig.warmup_steps, place,
                                         TrainTaskConfig.learning_rate)
    optimizer = fluid.optimizer.Adam(
        #learning_rate=lr_scheduler.learning_rate,
        learning_rate=TrainTaskConfig.learning_rate,
        beta1=TrainTaskConfig.beta1,
        beta2=TrainTaskConfig.beta2,
        epsilon=TrainTaskConfig.eps)

    #optimizer.minimize(cost)
    optimize_ops, params_grads = optimizer.minimize(cost)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program([cost])

    def test(exe):
        test_costs = []
        for batch_id, data in enumerate(test_reader()):
            if len(data) != args.batch_size:
                continue

            data_input = prepare_batch_input(
                data, encoder_input_data_names + decoder_input_data_names[:-1] +
                label_data_names, ModelHyperParams.src_pad_idx,
                ModelHyperParams.trg_pad_idx, ModelHyperParams.max_length,
                ModelHyperParams.n_head)

            test_cost = exe.run(test_program,
                                feed=data_input,
                                fetch_list=[cost])[0]

            test_costs.append(test_cost)
        return np.mean(test_costs)

    def train_loop(exe, trainer_prog):
        ts = time.time()
        for pass_id in xrange(args.pass_num):
            for batch_id, data in enumerate(train_reader()):
                print("batch_id:", batch_id)
                # The current program desc is coupled with batch_size, thus all
                # mini-batches must have the same number of instances currently.
                if len(data) != args.batch_size:
                    continue

                start_time = time.time()
                data_input = prepare_batch_input(
                    data, encoder_input_data_names + decoder_input_data_names[:-1] +
                    label_data_names, ModelHyperParams.src_pad_idx,
                    ModelHyperParams.trg_pad_idx, ModelHyperParams.max_length,
                    ModelHyperParams.n_head)

                #print("feed0:", data_input)
                #print("fetch_list0:", [cost])

                lr_scheduler.update_learning_rate(data_input)
                print("before exe run in train_loop")
                outs = exe.run(trainer_prog,
                           feed=data_input,
                           fetch_list=[cost],
                           use_program_cache=True)

                cost_val = np.array(outs[0])
                print("pass_id = %d batch = %d  cost = %f speed = %.2f sample/s" %
                      (pass_id, batch_id, cost_val, len(data) / (time.time() - start_time)))

            # Validate and save the model for inference.
            val_cost = test(exe)
            #pass_elapsed = time.time() - start_time
            #print("pass_id = " + str(pass_id) + " val_cost = " + str(val_cost))
            print("pass_id = %d batch = %d  cost = %f speed = %.2f sample/s" %
                  (pass_id, batch_id, cost_val, len(data) / (time.time() - ts)))

    if args.local:
        # Initialize the parameters.
        exe.run(fluid.framework.default_startup_program())
        #print("local start_up:")
        #print(debuger.pprint_program_codes(fluid.framework.default_startup_program()))
        for pos_enc_param_name in pos_enc_param_names:
            #print("pos_enc_param_name:", pos_enc_param_name)
            pos_enc_param = fluid.global_scope().find_var(
                pos_enc_param_name).get_tensor()
            pos_enc_param.set(
                position_encoding_init(ModelHyperParams.max_length + 1,
                                       ModelHyperParams.d_model), place)

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.wmt16.train(ModelHyperParams.src_vocab_size,
                                           ModelHyperParams.trg_vocab_size),
                buf_size=100000),
            batch_size=args.batch_size)

        test_reader = paddle.batch(
            paddle.dataset.wmt16.validation(ModelHyperParams.src_vocab_size,
                                            ModelHyperParams.trg_vocab_size),
            batch_size=args.batch_size)

        train_loop(exe, fluid.default_main_program())
    else:
        trainers = int(os.getenv("TRAINERS"))  # total trainer count
        print("trainers total: ", trainers)

        training_role = os.getenv(
            "TRAINING_ROLE",
            "TRAINER")  # get the training role: trainer/pserver

        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id=args.task_index,
            pservers=args.ps_hosts,
            trainers=trainers)

        if training_role == "PSERVER":
            current_endpoint = os.getenv("POD_IP") + ":" + os.getenv(
                "PADDLE_INIT_PORT")
            if not current_endpoint:
                print("need env SERVER_ENDPOINT")
                exit(1)
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            #print("cost 0:", cost)
            #print("before run start up")

            # Parameter initialization
            exe.run(fluid.default_startup_program())

            #print("cluster start_up:")
            #print(debuger.pprint_program_codes(fluid.framework.default_startup_program()))

            for pos_enc_param_name in pos_enc_param_names:
                #print("pos_enc_param_name:", pos_enc_param_name)
                pos_enc_param = fluid.global_scope().find_var(
                    pos_enc_param_name).get_tensor()
                pos_enc_param.set(
                    position_encoding_init(ModelHyperParams.max_length + 1,
                                           ModelHyperParams.d_model), place)

            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.wmt16.train(ModelHyperParams.src_vocab_size,
                                               ModelHyperParams.trg_vocab_size),
                    buf_size=100000),
                batch_size=args.batch_size)

            test_reader = paddle.batch(
                paddle.dataset.wmt16.validation(ModelHyperParams.src_vocab_size,
                                                ModelHyperParams.trg_vocab_size),
                batch_size=args.batch_size)

            #print("before get trainer program")
            trainer_prog = t.get_trainer_program()
            #print("before start")
            # feeder = fluid.DataFeeder(feed_list=[images, label], place=place)
            # TODO(typhoonzero): change trainer startup program to fetch parameters from pserver
            # exe.run(fluid.default_startup_program())

            train_loop(exe, trainer_prog)
        else:
            print("environment var TRAINER_ROLE should be TRAINER os PSERVER")

def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

if __name__ == "__main__":
    print_arguments()
    main()
