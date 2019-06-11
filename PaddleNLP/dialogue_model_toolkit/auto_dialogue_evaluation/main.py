"""
Auto dialogue evaluation task
"""

import os
import sys
import six
import numpy as np
import time
import multiprocessing
import paddle
import paddle.fluid as fluid
import reader as reader
import evaluation as eva
import init as init

try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3

sys.path.append('../../models/dialogue_model_toolkit/auto_dialogue_evaluation/')
from net import Network
import config

def train(args):
    """Train
    """
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    net = Network(args.vocab_size, args.emb_size, args.hidden_size)

    train_program = fluid.Program()
    train_startup = fluid.Program()
    if "CE_MODE_X" in os.environ:
        train_program.random_seed = 110
        train_startup.random_seed = 110
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            logits, loss = net.network(args.loss_type)
            loss.persistable = True
            logits.persistable = True
            # gradient clipping
            fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByValue(
                max=1.0, min=-1.0))

            optimizer = fluid.optimizer.Adam(
                learning_rate=args.learning_rate)
            optimizer.minimize(loss)
            print("begin memory optimization ...")
            fluid.memory_optimize(train_program)
            print("end memory optimization ...")

    test_program = fluid.Program()
    test_startup = fluid.Program()
    if "CE_MODE_X" in os.environ:
        test_program.random_seed = 110
        test_startup.random_seed = 110
    with fluid.program_guard(test_program, test_startup):
        with fluid.unique_name.guard():
            logits, loss = net.network(args.loss_type) 
            loss.persistable = True
            logits.persistable = True

    test_program = test_program.clone(for_test=True)
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    print("device count %d" % dev_count)
    print("theoretical memory usage: ")
    print(
        fluid.contrib.memory_usage(
            program=train_program, batch_size=args.batch_size))

    exe = fluid.Executor(place)
    exe.run(train_startup)
    exe.run(test_startup)

    train_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda, loss_name=loss.name, main_program=train_program)

    test_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda,
        main_program=test_program,
        share_vars_from=train_exe)

    if args.word_emb_init is not None:
        print("start loading word embedding init ...")
        if six.PY2:
            word_emb = np.array(pickle.load(open(args.word_emb_init,
                                                 'rb'))).astype('float32')
        else:
            word_emb = np.array(
                pickle.load(
                    open(args.word_emb_init, 'rb'), encoding="bytes")).astype(
                        'float32')
        net.set_word_embedding(word_emb, place)
        print("finish init word embedding  ...")

    print("start loading data ...")

    def train_with_feed(batch_data):
        """
        Train on one batch
        """
        #to do get_feed_names
        feed_dict = dict(zip(net.get_feed_names(), batch_data))

        cost = train_exe.run(feed=feed_dict, fetch_list=[loss.name])
        return cost[0]

    def test_with_feed(batch_data):
        """
        Test on one batch
        """
        feed_dict = dict(zip(net.get_feed_names(), batch_data))

        score = test_exe.run(feed=feed_dict, fetch_list=[logits.name])
        return score[0]

    def evaluate():
        """
        Evaluate to choose model
        """
        val_batches = reader.batch_reader(
            args.val_path, args.batch_size, place, args.max_len, 1)
        scores = []
        labels = []
        for batch in val_batches:
            scores.extend(test_with_feed(batch))
            labels.extend([x[0] for x in batch[2]])

        return eva.evaluate_Recall(zip(scores, labels))
    
    def save_exe(step, best_recall):
        """
        Save exe conditional
        """
        recall_dict = evaluate()
        print('evaluation recall result:')
        print('1_in_2: %s\t1_in_10: %s\t2_in_10: %s\t5_in_10: %s' % (
            recall_dict['1_in_2'], recall_dict['1_in_10'], 
            recall_dict['2_in_10'], recall_dict['5_in_10']))

        if recall_dict['1_in_10'] > best_recall and step != 0:
            fluid.io.save_inference_model(args.save_path, 
                net.get_feed_inference_names(), 
                logits, exe, main_program=train_program)

            print("Save model at step %d ... " % step)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            best_recall = recall_dict['1_in_10']
        return best_recall

    # train over different epoches
    global_step, train_time = 0, 0.0
    best_recall = 0 
    for epoch in six.moves.xrange(args.num_scan_data):
        train_batches = reader.batch_reader(
            args.train_path, args.batch_size, place, 
            args.max_len, args.sample_pro)

        begin_time = time.time()
        sum_cost = 0
        ce_cost = 0
        for batch in train_batches:
            if (args.save_path is not None) and (global_step % args.save_step == 0):
                best_recall = save_exe(global_step, best_recall)

            cost = train_with_feed(batch)
            global_step += 1
            sum_cost += cost.mean()
            ce_cost = cost.mean()

            if global_step % args.print_step == 0:
                print('training step %s avg loss %s' % (global_step, sum_cost / args.print_step))
                sum_cost = 0

        pass_time_cost = time.time() - begin_time
        train_time += pass_time_cost
        print("Pass {0}, pass_time_cost {1}"
              .format(epoch, "%2.2f sec" % pass_time_cost))
        if "CE_MODE_X" in os.environ and epoch == args.num_scan_data - 1:
            card_num = get_cards()
            print("kpis\ttrain_duration_card%s\t%s" % (card_num, pass_time_cost))
            print("kpis\ttrain_loss_card%s\t%s" % (card_num, ce_cost))


def finetune(args):
    """
    Finetune
    """
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    net = Network(args.vocab_size, args.emb_size, args.hidden_size)

    train_program = fluid.Program()
    train_startup = fluid.Program()
    if "CE_MODE_X" in os.environ:
        train_program.random_seed = 110
        train_startup.random_seed = 110
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            logits, loss = net.network(args.loss_type)
            loss.persistable = True
            logits.persistable = True
            # gradient clipping
            fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByValue(
                max=1.0, min=-1.0))

            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.exponential_decay(
                    learning_rate=args.learning_rate,
                    decay_steps=400,
                    decay_rate=0.9,
                    staircase=True))
            optimizer.minimize(loss)
            print("begin memory optimization ...")
            fluid.memory_optimize(train_program)
            print("end memory optimization ...")

    test_program = fluid.Program()
    test_startup = fluid.Program()
    if "CE_MODE_X" in os.environ:
        test_program.random_seed = 110
        test_startup.random_seed = 110
    with fluid.program_guard(test_program, test_startup):
        with fluid.unique_name.guard():
            logits, loss = net.network(args.loss_type) 
            loss.persistable = True
            logits.persistable = True

    test_program = test_program.clone(for_test=True)
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    print("device count %d" % dev_count)
    print("theoretical memory usage: ")
    print(
        fluid.contrib.memory_usage(
            program=train_program, batch_size=args.batch_size))

    exe = fluid.Executor(place)
    exe.run(train_startup)
    exe.run(test_startup)

    train_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda, loss_name=loss.name, main_program=train_program)

    test_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda,
        main_program=test_program,
        share_vars_from=train_exe)

    if args.init_model:
        init.init_pretraining_params(
            exe,
            args.init_model,
            main_program=train_startup)
        print('sccuess init %s' % args.init_model)

    print("start loading data ...")

    def train_with_feed(batch_data):
        """
        Train on one batch
        """
        #to do get_feed_names
        feed_dict = dict(zip(net.get_feed_names(), batch_data))

        cost = train_exe.run(feed=feed_dict, fetch_list=[loss.name])
        return cost[0]

    def test_with_feed(batch_data):
        """
        Test on one batch
        """
        feed_dict = dict(zip(net.get_feed_names(), batch_data))

        score = test_exe.run(feed=feed_dict, fetch_list=[logits.name])
        return score[0]

    def evaluate():
        """
        Evaluate to choose model
        """
        val_batches = reader.batch_reader(
            args.val_path, args.batch_size, place, args.max_len, 1)
        scores = []
        labels = []
        for batch in val_batches:
            scores.extend(test_with_feed(batch))
            labels.extend([x[0] for x in batch[2]])
        scores = [x[0] for x in scores]
        return eva.evaluate_cor(scores, labels)
    
    def save_exe(step, best_cor):
        """
        Save exe conditional
        """
        cor = evaluate()
        print('evaluation cor relevance %s' % cor)
        if cor > best_cor and step != 0:
            fluid.io.save_inference_model(args.save_path, 
                net.get_feed_inference_names(), logits, 
                exe, main_program=train_program)
            print("Save model at step %d ... " % step)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            best_cor = cor
        return best_cor

    # train over different epoches
    global_step, train_time = 0, 0.0
    best_cor = 0.0 
    pre_index = -1
    for epoch in six.moves.xrange(args.num_scan_data):
        train_batches = reader.batch_reader(
            args.train_path, 
            args.batch_size, place, 
            args.max_len, args.sample_pro)

        begin_time = time.time()
        sum_cost = 0
        for batch in train_batches:
            if (args.save_path is not None) and (global_step % args.save_step == 0):
                best_cor = save_exe(global_step, best_cor)

            cost = train_with_feed(batch)
            global_step += 1
            sum_cost += cost.mean()

            if global_step % args.print_step == 0:
                print('training step %s avg loss %s' % (global_step, sum_cost / args.print_step))
                sum_cost = 0

        pass_time_cost = time.time() - begin_time
        train_time += pass_time_cost
        print("Pass {0}, pass_time_cost {1}"
              .format(epoch, "%2.2f sec" % pass_time_cost))


def evaluate(args):
    """
    Evaluate model for both pretrained and finetuned 
    """
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    t0 = time.time()

    with fluid.scope_guard(fluid.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(
            args.init_model, exe)

        global_step, infer_time = 0, 0.0
        test_batches = reader.batch_reader(
            args.test_path, args.batch_size, place,
            args.max_len, 1)
        scores = []
        labels = []
        for batch in test_batches:
            logits = exe.run(
                infer_program,
                feed = {
                    'context_wordseq': batch[0],
                    'response_wordseq': batch[1]},
                fetch_list = fetch_vars)
            logits = [x[0] for x in logits[0]] 

            scores.extend(logits)
            labels.extend([x[0] for x in batch[2]])

        mean_score = sum(scores)/len(scores)
        if args.loss_type == 'CLS':
            recall_dict = eva.evaluate_Recall(zip(scores, labels))
            print('mean score: %s' % mean_score)
            print('evaluation recall result:')
            print('1_in_2: %s\t1_in_10: %s\t2_in_10: %s\t5_in_10: %s' % (
                recall_dict['1_in_2'], recall_dict['1_in_10'], 
                recall_dict['2_in_10'], recall_dict['5_in_10']))
        elif args.loss_type == 'L2':
            cor = eva.evaluate_cor(scores, labels)
            print('mean score: %s\nevaluation cor resuls:%s' % (mean_score, cor))
        else:
            raise ValueError

        t1 = time.time()
        print("finish evaluate model:%s on data:%s time_cost(s):%.2f" %
              (args.init_model, args.test_path, t1 - t0))


def infer(args):
    """
    Inference function 
    """
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    t0 = time.time()

    with fluid.scope_guard(fluid.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(
            args.init_model, exe)

        global_step, infer_time = 0, 0.0
        test_batches = reader.batch_reader(
            args.test_path, args.batch_size, place,
            args.max_len, 1)
        scores = []
        for batch in test_batches:
            logits = exe.run(
                infer_program,
                feed = {
                    'context_wordseq': batch[0],
                    'response_wordseq': batch[1]},
                fetch_list = fetch_vars)
            logits = [x[0] for x in logits[0]] 

            scores.extend(logits)

        in_file = open(args.test_path, 'r')
        out_path = args.test_path + '.infer'
        out_file = open(out_path, 'w')
        for line, s in zip(in_file, scores):
            out_file.write('%s\t%s\n' % (line.strip(), s))
            
        in_file.close()
        out_file.close()

        t1 = time.time()
        print("finish infer model:%s out file: %s time_cost(s):%.2f" %
              (args.init_model, out_path, t1 - t0))


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


def main():
    """
    main
    """
    args = config.parse_args()
    config.print_arguments(args)

    if args.do_train == True:
        if args.loss_type == 'CLS':
            train(args)
        elif args.loss_type == 'L2':
            finetune(args)
        else:
            raise ValueError
    elif args.do_val == True:
        evaluate(args)
    elif args.do_infer == True:
        infer(args)
    else:
        raise ValueError

if __name__ == '__main__':
    main()
