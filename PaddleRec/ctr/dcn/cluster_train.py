import argparse
import os
import sys
import time
from collections import OrderedDict

import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

from network import DCN


def parse_args():
    parser = argparse.ArgumentParser("dcn cluster train.")
    parser.add_argument(
        '--train_data_dir',
        type=str,
        default='dist_data/dist_train_data',
        help='The path of train data')
    parser.add_argument(
        '--test_valid_data_dir',
        type=str,
        default='dist_data/dist_test_valid_data',
        help='The path of test and valid data')
    parser.add_argument(
        '--vocab_dir',
        type=str,
        default='dist_data/vocab',
        help='The path of generated vocabs')
    parser.add_argument(
        '--cat_feat_num',
        type=str,
        default='dist_data/cat_feature_num.txt',
        help='The path of generated cat_feature_num.txt')
    parser.add_argument(
        '--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument('--num_epoch', type=int, default=10, help="train epoch")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help='The path for model to store')
    parser.add_argument(
        '--num_thread', type=int, default=1, help='The number of threads')
    parser.add_argument('--test_epoch', type=str, default='1')
    parser.add_argument(
        '--dnn_hidden_units',
        nargs='+',
        type=int,
        default=[1024, 1024],
        help='DNN layers and hidden units')
    parser.add_argument(
        '--cross_num',
        type=int,
        default=6,
        help='The number of Cross network layers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--l2_reg_cross',
        type=float,
        default=1e-5,
        help='Cross net l2 regularizer coefficient')
    parser.add_argument(
        '--use_bn',
        type=bool,
        default=True,
        help='Whether use batch norm in dnn part')
    parser.add_argument(
        '--is_sparse',
        action='store_true',
        required=False,
        default=False,
        help='embedding will use sparse or not, (default: False)')
    parser.add_argument(
        '--clip_by_norm', type=float, default=100.0, help="gradient clip norm")
    parser.add_argument('--print_steps', type=int, default=5)
    parser.add_argument('--use_gpu', type=int, default=1)

    # dist params
    parser.add_argument(
        '--load_model_dir',
        type=str,
        default=None,
        help='The path for saved model loading before training')
    parser.add_argument('--is_local', type=int, default=1, help='whether local')
    parser.add_argument(
        '--num_devices', type=int, default=1, help='Number of GPU devices')
    parser.add_argument(
        '--role', type=str, default='pserver', help='trainer or pserver')
    parser.add_argument(
        '--endpoints',
        type=str,
        default='127.0.0.1:6000',
        help='The pserver endpoints, like: 127.0.0.1:6000, 127.0.0.1:6001')
    parser.add_argument(
        '--current_endpoint',
        type=str,
        default='127.0.0.1:6000',
        help='The current_endpoint')
    parser.add_argument(
        '--trainer_id',
        type=int,
        default=0,
        help='trainer id ,only trainer_id=0 save model')
    parser.add_argument(
        '--trainers',
        type=int,
        default=1,
        help='The num of trianers, (default: 1)')
    args = parser.parse_args()
    return args


def train():
    """ do training """
    args = parse_args()
    print(args)

    endpoints = [ep.strip() for ep in args.endpoints.split(",")]
    if args.role.upper() == "PSERVER":
        current_id = endpoints.index(args.current_endpoint)
    else:
        current_id = args.trainer_id
    role = role_maker.UserDefinedRoleMaker(
        current_id=current_id,
        role=role_maker.Role.WORKER
        if args.role.upper() == "TRAINER" else role_maker.Role.SERVER,
        worker_num=args.trainers,
        server_endpoints=endpoints)

    is_first_trainer = False
    if args.role.upper() == "TRAINER" and current_id == 0:
        is_first_trainer = True

    if is_first_trainer and not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    fleet.init(role)

    strategy = DistributeTranspilerConfig()
    strategy.sync_mode = False

    cat_feat_dims_dict = OrderedDict()
    for line in open(args.cat_feat_num):
        spls = line.strip().split()
        assert len(spls) == 2
        cat_feat_dims_dict[spls[0]] = int(spls[1])

    dcn_model = DCN(args.cross_num, args.dnn_hidden_units, args.l2_reg_cross,
                    args.use_bn, args.clip_by_norm, cat_feat_dims_dict,
                    args.is_sparse)
    dcn_model.build_network()
    optimizer = fluid.optimizer.Adam(learning_rate=args.lr)
    if not args.is_local:
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(dcn_model.loss)

    def train_loop(main_program, startup_program):
        """ train network """
        start_time = time.time()
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

        if args.use_gpu == 1:
            exe = fluid.Executor(fluid.CUDAPlace(0))
            dataset.set_thread(1)
        else:
            exe = fluid.Executor(fluid.CPUPlace())
            dataset.set_thread(args.num_thread)
        exe.run(startup_program)

        for epoch_id in range(args.num_epoch):
            start = time.time()
            sys.stderr.write('\nepoch%d start ...\n' % (epoch_id + 1))
            exe.train_from_dataset(
                program=main_program,
                dataset=dataset,
                fetch_list=[
                    dcn_model.loss, dcn_model.avg_logloss, dcn_model.auc_var
                ],
                fetch_info=['total_loss', 'avg_logloss', 'auc'],
                debug=False,
                print_period=args.print_steps)
            model_dir = args.model_output_dir + '/epoch_' + str(epoch_id + 1)
            sys.stderr.write('epoch%d is finished and takes %f s\n' % (
                (epoch_id + 1), time.time() - start))
            if is_first_trainer:  # only trainer 0 save model
                print("save model in {}".format(model_dir))
                fluid.io.save_persistables(
                    executor=exe, dirname=model_dir, main_program=main_program)

        print("train time cost {:.4f}".format(time.time() - start_time))
        print("finish training")

    if args.is_local:
        print("run local training")
        train_loop(fluid.default_main_program(),
                   fluid.default_startup_program())
    else:
        print("run distribute training")
        if fleet.is_server():
            print("run pserver")
            if args.load_model_dir:
                print('load model from path %s' % args.load_model_dir)
            fleet.init_server(args.load_model_dir)
            fleet.run_server()
        elif fleet.is_worker():
            print("run trainer")
            fleet.init_worker()
            train_loop(fleet.main_program, fleet.startup_program)
            fleet.stop_worker()


if __name__ == "__main__":
    train()
