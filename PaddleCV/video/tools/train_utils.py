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


def test_without_pyreader(test_exe,
                          test_reader,
                          test_feeder,
                          test_fetch_list,
                          test_metrics,
                          log_interval=0):
    test_metrics.reset()
    for test_iter, data in enumerate(test_reader()):
        test_outs = test_exe.run(test_fetch_list, feed=test_feeder.feed(data))
        loss = np.array(test_outs[0])
        pred = np.array(test_outs[1])
        label = np.array(test_outs[-1])
        test_metrics.accumulate(loss, pred, label)
        if log_interval > 0 and test_iter % log_interval == 0:
            test_metrics.calculate_and_log_out(loss, pred, label, \
                  info = '[TEST] test_iter {} '.format(test_iter))
    test_metrics.finalize_and_log_out("[TEST] Finish")


def test_with_pyreader(test_exe,
                       test_pyreader,
                       test_fetch_list,
                       test_metrics,
                       log_interval=0):
    if not test_pyreader:
        logger.error("[TEST] get pyreader failed.")
    test_pyreader.start()
    test_metrics.reset()
    test_iter = 0
    try:
        while True:
            test_outs = test_exe.run(fetch_list=test_fetch_list)
            loss = np.array(test_outs[0])
            pred = np.array(test_outs[1])
            label = np.array(test_outs[-1])
            test_metrics.accumulate(loss, pred, label)
            if log_interval > 0 and test_iter % log_interval == 0:
                test_metrics.calculate_and_log_out(loss, pred, label, \
                  info = '[TEST] test_iter {} '.format(test_iter))
            test_iter += 1
    except fluid.core.EOFException:
        test_metrics.finalize_and_log_out("[TEST] Finish")
    finally:
        test_pyreader.reset()


def train_without_pyreader(exe, train_prog, train_exe, train_reader, train_feeder, \
                           train_fetch_list, train_metrics, epochs = 10, \
                           log_interval = 0, valid_interval = 0, save_dir = './', \
                           save_model_name = 'model', test_exe = None, test_reader = None, \
                           test_feeder = None, test_fetch_list = None, test_metrics = None):
    for epoch in range(epochs):
        log_lr_and_step()
        epoch_periods = []
        for train_iter, data in enumerate(train_reader()):
            cur_time = time.time()
            train_outs = train_exe.run(train_fetch_list,
                                       feed=train_feeder.feed(data))
            period = time.time() - cur_time
            epoch_periods.append(period)
            loss = np.array(train_outs[0])
            pred = np.array(train_outs[1])
            label = np.array(train_outs[-1])
            if log_interval > 0 and (train_iter % log_interval == 0):
                # eval here
                train_metrics.calculate_and_log_out(loss, pred, label, \
                       info = '[TRAIN] Epoch {}, iter {} '.format(epoch, train_iter))
            train_iter += 1
        logger.info('[TRAIN] Epoch {} training finished, average time: {}'.
                    format(epoch, np.mean(epoch_periods)))
        save_model(exe, train_prog, save_dir, save_model_name,
                   "_epoch{}".format(epoch))
        if test_exe and valid_interval > 0 and (epoch + 1
                                                ) % valid_interval == 0:
            test_without_pyreader(test_exe, test_reader, test_feeder,
                                  test_fetch_list, test_metrics, log_interval)



def train_with_pyreader(exe, train_prog, train_exe, train_pyreader, \
                        train_fetch_list, train_metrics, epochs = 10, \
                        log_interval = 0, valid_interval = 0, \
                        save_dir = './', save_model_name = 'model', \
                        test_exe = None, test_pyreader = None, \
                        test_fetch_list = None, test_metrics = None):
    if not train_pyreader:
        logger.error("[TRAIN] get pyreader failed.")
    for epoch in range(epochs):
        log_lr_and_step()
        train_pyreader.start()
        train_metrics.reset()
        try:
            train_iter = 0
            epoch_periods = []
            while True:
                cur_time = time.time()
                train_outs = train_exe.run(fetch_list=train_fetch_list)
                period = time.time() - cur_time
                epoch_periods.append(period)
                loss = np.array(train_outs[0])
                pred = np.array(train_outs[1])
                label = np.array(train_outs[-1])
                if log_interval > 0 and (train_iter % log_interval == 0):
                    # eval here
                    train_metrics.calculate_and_log_out(loss, pred, label, \
                                info = '[TRAIN] Epoch {}, iter {} '.format(epoch, train_iter))
                train_iter += 1
        except fluid.core.EOFException:
            # eval here
            logger.info('[TRAIN] Epoch {} training finished, average time: {}'.
                        format(epoch, np.mean(epoch_periods)))
            save_model(exe, train_prog, save_dir, save_model_name,
                       "_epoch{}".format(epoch))
            if test_exe and valid_interval > 0 and (epoch + 1
                                                    ) % valid_interval == 0:
                test_with_pyreader(test_exe, test_pyreader, test_fetch_list,
                                   test_metrics, log_interval)
        finally:
            epoch_period = []
            train_pyreader.reset()


def save_model(exe, program, save_dir, model_name, postfix=None):
    model_path = os.path.join(save_dir, model_name + postfix)
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    fluid.io.save_persistables(exe, model_path, main_program=program)
