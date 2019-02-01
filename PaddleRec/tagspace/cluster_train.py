import os
import sys
import time
import six
import numpy as np
import math
import argparse
import paddle
import paddle.fluid as fluid
import time
import utils
import net

SEED = 102

def parse_args():
    parser = argparse.ArgumentParser("TagSpace benchmark.")
    parser.add_argument(
        '--neg_size', type=int, default=3, help='neg/pos ratio')
    parser.add_argument(
        '--train_dir', type=str, default='train_data', help='train file address')
    parser.add_argument(
        '--vocab_text_path', type=str, default='vocab_text.txt', help='vocab_text file address')
    parser.add_argument(
        '--vocab_tag_path', type=str, default='vocab_tag.txt', help='vocab_text file address')
    parser.add_argument(
        '--is_local', type=int, default=1, help='whether local')
    parser.add_argument(
        '--model_dir', type=str, default='model_', help='model dir')
    parser.add_argument(
        '--batch_size', type=int, default=5, help='num of batch size')
    parser.add_argument(
        '--print_batch', type=int, default=10, help='num of print batch')
    parser.add_argument(
        '--pass_num', type=int, default=10, help='num of epoch')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether use gpu')
    parser.add_argument(
        '--base_lr', type=float, default=0.01, help='learning rate')
    parser.add_argument(
        '--num_devices', type=int, default=1, help='Number of GPU devices')
    parser.add_argument(
        '--role', type=str, default='pserver', help='trainer or pserver')
    parser.add_argument(
        '--endpoints', type=str, default='127.0.0.1:6000', help='The pserver endpoints, like: 127.0.0.1:6000, 127.0.0.1:6001')
    parser.add_argument(
        '--current_endpoint', type=str, default='127.0.0.1:6000', help='The current_endpoint')
    parser.add_argument(
        '--trainer_id', type=int, default=0, help='trainer id ,only trainer_id=0 save model')
    parser.add_argument(
        '--trainers', type=int, default=1, help='The num of trianers, (default: 1)')
    args = parser.parse_args()
    return args

def get_cards(args):
    return args.num_devices

def train():
    """ do training """
    args = parse_args()
    train_dir = args.train_dir
    vocab_text_path = args.vocab_text_path
    vocab_tag_path = args.vocab_tag_path
    use_cuda = True if args.use_cuda else False
    batch_size = args.batch_size
    neg_size = args.neg_size
    vocab_text_size, vocab_tag_size, train_reader = utils.prepare_data(
        file_dir=train_dir, vocab_text_path=vocab_text_path, 
        vocab_tag_path=vocab_tag_path, neg_size=neg_size, 
        batch_size=batch_size * get_cards(args), 
        buffer_size=batch_size*100, is_train=True)
    """ train network """
    # Train program
    avg_cost, correct, cos_pos = net.network(vocab_text_size, vocab_tag_size, neg_size=neg_size)

    # Optimization to minimize lost
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=args.base_lr)
    sgd_optimizer.minimize(avg_cost)

    def train_loop(main_program):
        # Initialize executor
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        pass_num = args.pass_num
        model_dir = args.model_dir
        fetch_list = [avg_cost.name]
        exe.run(fluid.default_startup_program())
        total_time = 0.0
        for pass_idx in range(pass_num):
            epoch_idx = pass_idx + 1
            print("epoch_%d start" % epoch_idx)
            t0 = time.time()
            for batch_id, data in enumerate(train_reader()):
                lod_text_seq = utils.to_lodtensor([dat[0] for dat in data], place)
                lod_pos_tag = utils.to_lodtensor([dat[1] for dat in data], place)
                lod_neg_tag = utils.to_lodtensor([dat[2] for dat in data], place)
                loss_val, correct_val = exe.run(
                        feed={
                            "text": lod_text_seq,
                            "pos_tag": lod_pos_tag,
                            "neg_tag": lod_neg_tag},
                        fetch_list=[avg_cost.name, correct.name])
                if batch_id % args.print_batch == 0:
                    print("TRAIN --> pass: {} batch_num: {} avg_cost: {}, acc: {}"
                            .format(pass_idx, (batch_id+10) * batch_size, np.mean(loss_val),
                                    float(np.sum(correct_val)) / batch_size))
            t1 = time.time()
            total_time += t1 - t0
            print("epoch:%d num_steps:%d time_cost(s):%f" %
                  (epoch_idx, batch_id, total_time / epoch_idx))
            save_dir = "%s/epoch_%d" % (model_dir, epoch_idx)
            feed_var_names = ["text", "pos_tag"]
            fetch_vars = [cos_pos]
            fluid.io.save_inference_model(save_dir, feed_var_names, fetch_vars, exe)
        print("finish training")
    
    if args.is_local:
        print("run local training")
        train_loop(fluid.default_main_program())
    else:
        print("run distribute training")
        t = fluid.DistributeTranspiler()
        t.transpile(args.trainer_id, pservers=args.endpoints, trainers=args.trainers)
        if args.role == "pserver":
            print("run psever")
            pserver_prog = t.get_pserver_program(args.current_endpoint)
            pserver_startup = t.get_startup_program(args.current_endpoint,
                                                    pserver_prog)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif args.role == "trainer":
            print("run trainer")
            train_loop(t.get_trainer_program())

if __name__ == "__main__":
    train()
