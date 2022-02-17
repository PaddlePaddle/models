from __future__ import print_function, absolute_import, division
import os
import random
import sys
import time
from collections import OrderedDict

import paddle.fluid as fluid

from config import parse_args
from network import DCN
import utils
"""
train DCN model
"""


def train(args):
    """train and save DCN model

    :param args: hyperparams of model
    :return:
    """
    # ce
    if args.enable_ce:
        SEED = 102
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

    cat_feat_dims_dict = OrderedDict()
    for line in open(args.cat_feat_num):
        spls = line.strip().split()
        assert len(spls) == 2
        cat_feat_dims_dict[spls[0]] = int(spls[1])
    dcn_model = DCN(args.cross_num, args.dnn_hidden_units, args.l2_reg_cross,
                    args.use_bn, args.clip_by_norm, cat_feat_dims_dict,
                    args.is_sparse)
    dcn_model.build_network()
    dcn_model.backward(args.lr)

    # config dataset
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(dcn_model.data_list)
    pipe_command = 'python reader.py {}'.format(args.vocab_dir)
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(args.batch_size)
    dataset.set_thread(args.num_thread)
    train_filelist = [
        os.path.join(args.train_data_dir, fname)
        for fname in next(os.walk(args.train_data_dir))[2]
    ]
    dataset.set_filelist(train_filelist)
    num_epoch = args.num_epoch
    if args.steps:
        epoch = args.steps * args.batch_size / 41000000
        full_epoch = int(epoch // 1)
        last_epoch = epoch % 1
        train_filelists = [train_filelist for _ in range(full_epoch)] + [
            random.sample(train_filelist, int(
                len(train_filelist) * last_epoch))
        ]
        num_epoch = full_epoch + 1
    print("train epoch: {}".format(num_epoch))

    # Executor
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    for epoch_id in range(num_epoch):
        start = time.time()
        sys.stderr.write('\nepoch%d start ...\n' % (epoch_id + 1))
        dataset.set_filelist(train_filelists[epoch_id])
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=dataset,
            fetch_list=[
                dcn_model.loss, dcn_model.avg_logloss, dcn_model.auc_var
            ],
            fetch_info=['total_loss', 'avg_logloss', 'auc'],
            debug=False,
            print_period=args.print_steps)
        model_dir = os.path.join(args.model_output_dir,
                                 'epoch_' + str(epoch_id + 1), "checkpoint")
        sys.stderr.write('epoch%d is finished and takes %f s\n' % (
            (epoch_id + 1), time.time() - start))
        fluid.save(fluid.default_main_program(), model_dir)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    args = parse_args()
    print(args)
    utils.check_version()
    train(args)
