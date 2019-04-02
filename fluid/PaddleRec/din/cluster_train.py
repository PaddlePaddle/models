import sys
import logging
import time
import numpy as np
import argparse
import paddle.fluid as fluid
import paddle
import time
import network
import reader
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("din")
    parser.add_argument(
        '--config_path',
        type=str,
        default='data/config.txt',
        help='dir of config')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='data/paddle_train.txt',
        help='dir of train file')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='din_amazon/',
        help='dir of saved model')
    parser.add_argument(
        '--batch_size', type=int, default=16, help='number of batch size')
    parser.add_argument(
        '--epoch_num', type=int, default=200, help='number of epoch')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether to use gpu')
    parser.add_argument(
        '--parallel',
        type=int,
        default=0,
        help='whether to use parallel executor')
    parser.add_argument(
        '--base_lr', type=float, default=0.85, help='based learning rate')
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
    args = parse_args()

    config_path = args.config_path
    train_path = args.train_dir
    epoch_num = args.epoch_num
    use_cuda = True if args.use_cuda else False
    use_parallel = True if args.parallel else False

    logger.info("reading data begins")
    user_count, item_count, cat_count = reader.config_read(config_path)
    #data_reader, max_len = reader.prepare_reader(train_path, args.batch_size)
    logger.info("reading data completes")

    avg_cost, pred = network.network(item_count, cat_count, 433)
    #fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
    base_lr = args.base_lr
    boundaries = [410000]
    values = [base_lr, 0.2]
    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=boundaries, values=values))
    sgd_optimizer.minimize(avg_cost)

    def train_loop(main_program):
        data_reader, max_len = reader.prepare_reader(train_path,
                                                     args.batch_size)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        feeder = fluid.DataFeeder(
            feed_list=[
                "hist_item_seq", "hist_cat_seq", "target_item", "target_cat",
                "label", "mask", "target_item_seq", "target_cat_seq"
            ],
            place=place)
        if use_parallel:
            train_exe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                loss_name=avg_cost.name,
                main_program=main_program)
        else:
            train_exe = exe
        logger.info("train begins")
        global_step = 0
        PRINT_STEP = 1000

        start_time = time.time()
        loss_sum = 0.0
        for id in range(epoch_num):
            epoch = id + 1
            for data in data_reader():
                global_step += 1
                results = train_exe.run(main_program,
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost.name, pred.name],
                                        return_numpy=True)
                loss_sum += results[0].mean()

                if global_step % PRINT_STEP == 0:
                    logger.info(
                        "epoch: %d\tglobal_step: %d\ttrain_loss: %.4f\t\ttime: %.2f"
                        % (epoch, global_step, loss_sum / PRINT_STEP,
                           time.time() - start_time))
                    start_time = time.time()
                    loss_sum = 0.0

                    if (global_step > 400000 and
                            global_step % PRINT_STEP == 0) or (
                                global_step < 400000 and
                                global_step % 50000 == 0):
                        save_dir = args.model_dir + "/global_step_" + str(
                            global_step)
                        feed_var_name = [
                            "hist_item_seq", "hist_cat_seq", "target_item",
                            "target_cat", "label", "mask", "target_item_seq",
                            "target_cat_seq"
                        ]
                        fetch_vars = [avg_cost, pred]
                        fluid.io.save_inference_model(save_dir, feed_var_name,
                                                      fetch_vars, exe)
        train_exe.close()

    t = fluid.DistributeTranspiler()
    t.transpile(
        args.trainer_id, pservers=args.endpoints, trainers=args.trainers)
    if args.role == "pserver":
        logger.info("run psever")
        prog, startup = t.get_pserver_programs(args.current_endpoint)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup)
        exe.run(prog)
    elif args.role == "trainer":
        logger.info("run trainer")
        train_loop(t.get_trainer_program())


if __name__ == "__main__":
    train()
