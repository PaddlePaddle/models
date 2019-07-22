import os
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import logging
import shutil

logger = logging.getLogger(__name__)


def log_lr_and_step():
    try:
        # In optimizers, if learning_rate is set as constant, lr_var
        # name is 'learning_rate_0', and iteration counter is not 
        # recorded. If learning_rate is set as decayed values from 
        # learning_rate_scheduler, lr_var name is 'learning_rate', 
        # and iteration counter is recorded with name '@LR_DECAY_COUNTER@', 
        # better impliment is required here
        lr_var = fluid.global_scope().find_var("learning_rate")
        if not lr_var:
            lr_var = fluid.global_scope().find_var("learning_rate_0")
        lr = np.array(lr_var.get_tensor())

        lr_count = '[-]'
        lr_count_var = fluid.global_scope().find_var("@LR_DECAY_COUNTER@")
        if lr_count_var:
            lr_count = np.array(lr_count_var.get_tensor())
        logger.info("------- learning rate {}, learning rate counter {} -----"
                    .format(np.array(lr), np.array(lr_count)))
    except:
        logger.warn("Unable to get learning_rate and LR_DECAY_COUNTER.")


def test_with_pyreader(exe,
                       compiled_test_prog,
                       test_pyreader,
                       test_fetch_list,
                       test_metrics,
                       log_interval=0,
                       save_model_name=''):
    if not test_pyreader:
        logger.error("[TEST] get pyreader failed.")
    test_pyreader.start()
    test_metrics.reset()
    test_iter = 0
    try:
        while True:
            #test_outs = test_exe.run(fetch_list=test_fetch_list)
            test_outs = exe.run(compiled_test_prog, fetch_list=test_fetch_list)
            test_metrics.accumulate(test_outs)
            if log_interval > 0 and test_iter % log_interval == 0:
                test_metrics.calculate_and_log_out(test_outs, \
                   info = '[TEST] test_iter {} '.format(test_iter))
            test_iter += 1
    except fluid.core.EOFException:
        test_metrics.finalize_and_log_out("[TEST] Finish")
    finally:
        test_pyreader.reset()


def train_with_pyreader(exe, train_prog, compiled_train_prog, train_pyreader, \
                        train_fetch_list, train_metrics, epochs = 10, \
                        log_interval = 0, valid_interval = 0, save_dir = './', \
                        save_model_name = 'model', enable_ce = False, \
                        compiled_test_prog = None, test_pyreader = None, \
                        test_fetch_list = None, test_metrics = None):
    if not train_pyreader:
        logger.error("[TRAIN] get pyreader failed.")
    epoch_periods = []
    train_loss = 0
    for epoch in range(epochs):
        log_lr_and_step()
        train_pyreader.start()
        train_metrics.reset()
        try:
            train_iter = 0
            epoch_periods = []
            while True:
                cur_time = time.time()
                #train_outs = train_exe.run(fetch_list=train_fetch_list)
                train_outs = exe.run(compiled_train_prog,
                                     fetch_list=train_fetch_list)
                period = time.time() - cur_time
                epoch_periods.append(period)
                if log_interval > 0 and (train_iter % log_interval == 0):
                    train_metrics.calculate_and_log_out(train_outs, \
                                info = '[TRAIN] Epoch {}, iter {} '.format(epoch, train_iter))
                train_iter += 1
        except fluid.core.EOFException:
            # eval here
            logger.info('[TRAIN] Epoch {} training finished, average time: {}'.
                        format(epoch, np.mean(epoch_periods[1:])))
            save_model(exe, train_prog, save_dir, save_model_name,
                       "_epoch{}".format(epoch))
            if test_exe and valid_interval > 0 and (epoch + 1
                                                    ) % valid_interval == 0:
                test_with_pyreader(exe, compiled_test_prog, test_pyreader,
                                   test_fetch_list, test_metrics, log_interval,
                                   save_model_name)
        finally:
            epoch_period = []
            train_pyreader.reset()
    #only for ce
    if enable_ce:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        gpu_num = len(cards.split(","))
        print("kpis\ttrain_cost_card{}\t{}".format(gpu_num, train_loss))
        print("kpis\ttrain_speed_card{}\t{}".format(gpu_num,
                                                    np.mean(epoch_periods)))


def save_model(exe, program, save_dir, model_name, postfix=None):
    model_path = os.path.join(save_dir, model_name + postfix)
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    fluid.io.save_persistables(exe, model_path, main_program=program)
