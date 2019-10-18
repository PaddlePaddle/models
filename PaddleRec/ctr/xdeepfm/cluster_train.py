import argparse
import os
import sys
import time
import network_conf

import paddle.fluid as fluid
import utils


def parse_args():
    parser = argparse.ArgumentParser("xdeepfm cluster train.")

    parser.add_argument(
        '--train_data_dir',
        type=str,
        default='data/train_data',
        help='The path of train data (default: data/train_data)')
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='data/test_data',
        help='The path of test data (default: models)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="The size of mini-batch (default:100)")
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=10,
        help="The number of epochs to train (default: 10)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        required=True,
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--num_thread',
        type=int,
        default=1,
        help='The number of threads (default: 1)')
    parser.add_argument('--test_epoch', type=str, default='1')
    parser.add_argument(
        '--layer_sizes_dnn',
        nargs='+',
        type=int,
        default=[10, 10, 10],
        help='The size of each layers')
    parser.add_argument(
        '--layer_sizes_cin',
        nargs='+',
        type=int,
        default=[10, 10],
        help='The size of each layers')
    parser.add_argument(
        '--act',
        type=str,
        default='relu',
        help='The activation of each layers (default: relu)')
    parser.add_argument(
        '--lr', type=float, default=1e-1, help='Learning rate (default: 1e-4)')
    parser.add_argument(
        '--reg', type=float, default=1e-4, help=' (default: 1e-4)')
    parser.add_argument('--num_field', type=int, default=39)
    parser.add_argument('--num_feat', type=int, default=28651)
    parser.add_argument(
        '--model_name',
        type=str,
        default='ctr_xdeepfm_model',
        help='The name of model (default: ctr_xdeepfm_model)')
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--print_steps', type=int, default=50)
    parser.add_argument('--is_local', type=int, default=1, help='whether local')
    parser.add_argument(
        '--is_sparse',
        action='store_true',
        required=False,
        default=False,
        help='embedding will use sparse or not, (default: False)')

    # dist params
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

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    loss, auc, data_list = eval('network_conf.' + args.model_name)(
        args.embedding_size, args.num_field, args.num_feat,
        args.layer_sizes_dnn, args.act, args.reg, args.layer_sizes_cin,
        args.is_sparse)
    optimizer = fluid.optimizer.SGD(
        learning_rate=args.lr,
        regularization=fluid.regularizer.L2DecayRegularizer(args.reg))
    optimizer.minimize(loss)

    def train_loop(main_program):
        """ train network """
        start_time = time.time()
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(data_list)
        dataset.set_pipe_command('python criteo_reader.py')
        dataset.set_batch_size(args.batch_size)
        dataset.set_filelist([
            os.path.join(args.train_data_dir, x)
            for x in os.listdir(args.train_data_dir)
        ])

        if args.use_gpu == 1:
            exe = fluid.Executor(fluid.CUDAPlace(0))
            dataset.set_thread(1)
        else:
            exe = fluid.Executor(fluid.CPUPlace())
            dataset.set_thread(args.num_thread)
        exe.run(fluid.default_startup_program())

        for epoch_id in range(args.num_epoch):
            start = time.time()
            sys.stderr.write('\nepoch%d start ...\n' % (epoch_id + 1))
            exe.train_from_dataset(
                program=main_program,
                dataset=dataset,
                fetch_list=[loss, auc],
                fetch_info=['loss', 'auc'],
                debug=False,
                print_period=args.print_steps)
            model_dir = os.path.join(args.model_output_dir,
                                     'epoch_' + str(epoch_id + 1))
            sys.stderr.write('epoch%d is finished and takes %f s\n' % (
                (epoch_id + 1), time.time() - start))
            if args.trainer_id == 0:  # only trainer 0 save model
                print("save model in {}".format(model_dir))
                fluid.io.save_persistables(
                    executor=exe, dirname=model_dir, main_program=main_program)

        print("train time cost {:.4f}".format(time.time() - start_time))
        print("finish training")

    if args.is_local:
        print("run local training")
        train_loop(fluid.default_main_program())
    else:
        print("run distribute training")
        t = fluid.DistributeTranspiler()
        t.transpile(
            args.trainer_id, pservers=args.endpoints, trainers=args.trainers)
        if args.role == "pserver":
            print("run psever")
            pserver_prog, pserver_startup = t.get_pserver_programs(
                args.current_endpoint)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif args.role == "trainer":
            print("run trainer")
            train_loop(t.get_trainer_program())


if __name__ == "__main__":
    utils.check_version()
    train()
