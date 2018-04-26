import os
import time
import argparse
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

from model import transformer, position_encoding_init
import model
from optim import LearningRateScheduler
from config import TrainTaskConfig, ModelHyperParams, pos_enc_param_names, \
        encoder_input_data_names, decoder_input_data_names, label_data_names
import paddle.fluid.debuger as debuger
import nist_data_provider
import sys
import data_util

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
    '--save_graph', type=str2bool, default=False, help="Save graph of network.")

parser.add_argument(
    '--exit_batch_id', type=int, default=200, help="Program exits when batch_id==exit_batch_id.")

parser.add_argument(
    '--learning_rate',
    type=float,
    default=TrainTaskConfig.learning_rate,
    help="Learning rate for training.")

parser.add_argument(
    '--device',
    type=str,
    default='GPU',
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

parser.add_argument(
    "--test_save",
    type=str2bool,
    default=False,
    help="test save model")


parser.add_argument(
    "--model_path",
    type=str,
    default="/pfs/dlnel/home/work-beijing-163-com/nmt",
    help="model path")

# Flags for defining the tf.train.Server
parser.add_argument(
    "--task_index", type=int, default=0, help="Index of task within the job")
args = parser.parse_args()

TrainTaskConfig.batch_size = args.batch_size

def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True):
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
            range(1, len(inst) + 1) + [0] * (max_len - len(inst))
            for inst in insts
        ])
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
                        n_head, d_model):
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    # These shape tensors are used in reshape_op.
    src_data_shape = np.array([len(insts), src_max_len, d_model], dtype="int32")
    trg_data_shape = np.array([len(insts), trg_max_len, d_model], dtype="int32")
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

    lbl_word, lbl_weight = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False)

    input_dict = dict(
        zip(input_data_names, [
            src_word, src_pos, src_slf_attn_bias, src_data_shape,
            src_slf_attn_pre_softmax_shape, src_slf_attn_post_softmax_shape,
            trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias,
            trg_data_shape, trg_slf_attn_pre_softmax_shape,
            trg_slf_attn_post_softmax_shape, trg_src_attn_pre_softmax_shape,
            trg_src_attn_post_softmax_shape, lbl_word, lbl_weight
        ]))
    return input_dict


def get_var(name,value):
    return fluid.layers.create_global_var(
            name=name,
            shape=[1],
            value=float(value),
            dtype="float32",
            persistable=True)

#@profile
def main():
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(
        args.device_id)
    exe = fluid.Executor(place)

    sum_cost, avg_cost, predict, token_num = transformer(
        ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size,
        ModelHyperParams.max_length + 1, ModelHyperParams.n_layer,
        ModelHyperParams.n_head, ModelHyperParams.d_key,
        ModelHyperParams.d_value, ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid, ModelHyperParams.dropout)

    '''
    lr_scheduler = LearningRateScheduler(ModelHyperParams.d_model,
                                         TrainTaskConfig.warmup_steps, place,
                                         TrainTaskConfig.learning_rate)
    '''

    warmup_steps = get_var("warmup_steps", value=TrainTaskConfig.warmup_steps)
    d_model = get_var("d_model", value=ModelHyperParams.d_model)

    lr_decay = fluid.layers\
        .learning_rate_scheduler\
        .noam_decay(d_model, warmup_steps)

    optimizer = fluid.optimizer.Adam(
        learning_rate = lr_decay,
        beta1=TrainTaskConfig.beta1,
        beta2=TrainTaskConfig.beta2,
        epsilon=TrainTaskConfig.eps)
    optimize_ops, params_grads = optimizer.minimize(avg_cost if TrainTaskConfig.use_avg_cost else sum_cost)


    # Program to do validation.
    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program([avg_cost])

    def test(exe):
        test_total_cost = 0
        test_total_token = 0
        for batch_id, data in enumerate(test_reader()):
            data_input = prepare_batch_input(
                data, encoder_input_data_names + decoder_input_data_names[:-1] +
                label_data_names, ModelHyperParams.eos_idx,
                ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)
            test_sum_cost, test_token_num = exe.run(
                inference_program,
                feed=data_input,
                fetch_list=[sum_cost, token_num],
                use_program_cache=True)
            test_total_cost += test_sum_cost
            test_total_token += test_token_num
        test_avg_cost = test_total_cost / test_total_token
        test_ppl = np.exp([min(test_avg_cost, 100)])
        return test_avg_cost, test_ppl

    '''
    def train_loop(exe, trainer_prog):
        # Initialize the parameters.
        """
        exe.run(fluid.framework.default_startup_program())
        """
        for pos_enc_param_name in pos_enc_param_names:
            pos_enc_param = fluid.global_scope().find_var(
                pos_enc_param_name).get_tensor()
            pos_enc_param.set(
                position_encoding_init(ModelHyperParams.max_length + 1,
                                       ModelHyperParams.d_model), place)
    '''

    def train_loop(exe, trainer_prog):
        for pass_id in xrange(args.pass_num):
            ts = time.time()
            total = 0
            pass_start_time = time.time()
            #print len(train_reader)
            for batch_id, data in enumerate(train_reader):
                #print len(data)
                if len(data) != args.batch_size:
                    continue

                total += len(data)
                start_time = time.time()
                data_input = prepare_batch_input(
                    data, encoder_input_data_names + decoder_input_data_names[:-1] +
                    label_data_names, ModelHyperParams.eos_idx,
                    ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                    ModelHyperParams.d_model)
                '''
                if args.local:
                    lr_scheduler.update_learning_rate(data_input)
                '''
                outs = exe.run(trainer_prog,
                               feed=data_input,
                               fetch_list=[sum_cost, avg_cost],
                               use_program_cache=True)
                sum_cost_val, avg_cost_val = np.array(outs[0]), np.array(outs[1])
                print("epoch: %d, batch: %d, sum loss: %f, avg loss: %f, ppl: %f, speed: %.2f" %
                      (pass_id, batch_id, sum_cost_val, avg_cost_val,
                       np.exp([min(avg_cost_val[0], 100)]), 
                       len(data) / (time.time() - start_time)))

                if args.test_save:
                    if batch_id == args.exit_batch_id:
                        print("batch_id: %d exit!" % batch_id)
                        break

            # Validate and save the model for inference.
            # val_avg_cost, val_ppl = test(exe)
            val_avg_cost, val_ppl = 0,0
            pass_end_time = time.time()
            time_consumed = pass_end_time - pass_start_time
            print("pass_id = %s time_consumed = %s val_avg_cost=%f val_ppl=%f speed: %.2f" % \
                  (str(pass_id), str(time_consumed), \
                     val_avg_cost, val_ppl, total / (time.time() - ts)))

            fluid.io.save_inference_model(
                os.path.join(args.model_path,
                             "pass_" + str(pass_id) + "_" + str(args.task_index) + ".infer.model"),
                encoder_input_data_names + decoder_input_data_names[:-1],
                [predict], exe)

            if args.test_save:
                break

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
                nist_data_provider.train("data", ModelHyperParams.src_vocab_size,
                                         ModelHyperParams.trg_vocab_size),
                buf_size=100000),
            batch_size=args.batch_size)

        '''
        test_reader = paddle.batch(
                nist_data_provider.train("data", ModelHyperParams.src_vocab_size,
                                         ModelHyperParams.trg_vocab_size),
            batch_size=TrainTaskConfig.batch_size)
        '''

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

            if args.save_graph: 
                block_no=0
                for t in pserver_startup.blocks: 
                    block_name="pserver_startup_block_%04d" % block_no
                    print block_name
                    print(debuger.draw_block_graphviz(t, path="./" + block_name+".dot"))
                    block_no+=1

                block_no=0
                for t in pserver_prog.blocks:
                    block_name="pserver_prog_block_%04d" % block_no
                    print(debuger.draw_block_graphviz(t, path="./" + block_name+".dot"))
                    block_no+=1

            print "begin run"
            exe.run(pserver_startup, save_program_to_file="./pserver_startup.desc")
            exe.run(pserver_prog, save_program_to_file="./pserver_loop.desc")
        elif training_role == "TRAINER":
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

            #print "/root/data/nist06n/data-%d/part-*" % (args.task_index),
            train_reader = data_util.DataLoader(
                src_vocab_fpath="/root/data/nist06n/cn_30001.dict",
                trg_vocab_fpath="/root/data/nist06n/en_30001.dict",
                fpattern="/root/data/nist06n/data-%d/part-*" % (args.task_index),
                batch_size=args.batch_size,
                token_batch_size=TrainTaskConfig.token_batch_size,
                sort_by_length=TrainTaskConfig.sort_by_length,
                shuffle=True)

            '''
            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    nist_data_provider.train("data", ModelHyperParams.src_vocab_size,
                                             ModelHyperParams.trg_vocab_size),
                    buf_size=100000),
                batch_size=args.batch_size)

            test_reader = paddle.batch(
                    nist_data_provider.train("data", ModelHyperParams.src_vocab_size,
                                             ModelHyperParams.trg_vocab_size),
                batch_size=TrainTaskConfig.batch_size)
            '''

            trainer_prog = t.get_trainer_program()
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

