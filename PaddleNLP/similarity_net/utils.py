# --coding=utf-8
"""
SimNet utilities.
"""

import time
import sys
import re
import os
import six
import numpy as np
import logging
import logging.handlers

"""
******functions for file processing******
"""


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    if not os.path.isfile(file_path):
        raise ValueError("vocabulary dose not exist under %s" % file_path)
    with open(file_path, 'r') as f:
        for line in f:
            items = line.strip('\n').split("\t")
            if items[0] not in vocab:
                vocab[items[0]] = int(items[1])
    vocab["<unk>"] = 0
    return vocab


def get_result_file(args):
    """
    Get Result File
    Args:
      conf_dict: Input path config
      samples_file_path: Data path of real training
      predictions_file_path: Prediction results path
    Returns:
      result_file: merge sample and predict result

    """
    with open(args.test_data_dir, "r") as test_file:
        with open("predictions.txt", "r") as predictions_file:
            with open(args.test_result_path, "w") as test_result_file:
                test_datas = [line.strip("\n") for line in test_file]
                predictions = [line.strip("\n") for line in predictions_file]
                for test_data, prediction in zip(test_datas, predictions):
                    test_result_file.write(test_data + "\t" + prediction + "\n")
    os.remove("predictions.txt")


"""
******functions for string processing******
"""


def pattern_match(pattern, line):
    """
    Check whether a string is matched
    Args:
      pattern: mathing pattern
      line : input string
    Returns:
      True/False
    """
    if re.match(pattern, line):
        return True
    else:
        return False


"""
******functions for parameter processing******
"""


def print_progress(task_name, percentage, style=0):
    """
    Print progress bar
    Args:
      task_name: The name of the current task
      percentage: Current progress
      style: Progress bar form
    """
    styles = ['#', '█']
    mark = styles[style] * percentage
    mark += ' ' * (100 - percentage)
    status = '%d%%' % percentage if percentage < 100 else 'Finished'
    sys.stdout.write('%+20s [%s] %s\r' % (task_name, mark, status))
    sys.stdout.flush()
    time.sleep(0.002)


def display_args(name, args):
    """
    Print parameter information
    Args:
      name: logger instance name
      args: Input parameter dictionary
    """
    logger = logging.getLogger(name)
    logger.info("The arguments passed by command line is :")
    for k, v in sorted(v for v in vars(args).items()):
        logger.info("{}:\t{}".format(k, v))


def import_class(module_path, module_name, class_name):
    """
    Load class dynamically
    Args:
      module_path: The current path of the module
      module_name: The module name
      class_name: The name of class in the import module
    Return:
      Return the attribute value of the class object
    """
    if module_path:
        sys.path.append(module_path)
    module = __import__(module_name)
    return getattr(module, class_name)


def str2bool(v):
    """
    String to Boolean
    """
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    Argument Class
    """

    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """
        Add argument
        """
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args):
    """
    Print Arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def init_log(log_path, level=logging.INFO, when="D", backup=7,
             format="%(levelname)s: %(asctime)s - %(filename)s:%(lineno)d * %(thread)d %(message)s",
             datefmt=None):
    """
    init_log - initialize log module

    Args:
      log_path      - Log file path prefix.
                      Log data will go to two files: log_path.log and log_path.log.wf
                      Any non-exist parent directories will be created automatically
      level         - msg above the level will be displayed
                      DEBUG < INFO < WARNING < ERROR < CRITICAL
                      the default value is logging.INFO
      when          - how to split the log file by time interval
                      'S' : Seconds
                      'M' : Minutes
                      'H' : Hours
                      'D' : Days
                      'W' : Week day
                      default value: 'D'
      format        - format of the log
                      default format:
                      %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
                      INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
      backup        - how many backup file to keep
                      default value: 7

    Raises:
        OSError: fail to create log directories
        IOError: fail to open log file
    """
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)

    # console Handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    logger.addHandler(consoleHandler)

    dir = os.path.dirname(log_path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    handler = logging.handlers.TimedRotatingFileHandler(log_path + ".log",
                                                        when=when,
                                                        backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.handlers.TimedRotatingFileHandler(log_path + ".log.wf",
                                                        when=when,
                                                        backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def set_level(level):
    """
    Reak-time set log level
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logging.info('log level is set to : %d' % level)


def get_level():
    """
    get Real-time log level
    """
    logger = logging.getLogger()
    return logger.level


def get_accuracy(preds, labels, mode, lamda=0.91):
    """
    compute accuracy
    """
    if mode == "pairwise":
        preds = np.array(list(map(lambda x: 1 if x[1] >= lamda else 0, preds)))
    else:
        preds = np.array(list(map(lambda x: np.argmax(x), preds)))
    labels = np.squeeze(labels)
    return np.mean(preds == labels)


def get_softmax(preds):
    """
    compute sotfmax
    """
    _exp = np.exp(preds)
    return _exp / np.sum(_exp, axis=1, keepdims=True)


def get_sigmoid(preds):
    """
    compute sigmoid
    """
    return 1 / (1 + np.exp(-preds))


def deal_preds_of_mmdnn(conf_dict, preds):
    """
    deal preds of mmdnn
    """
    if conf_dict['task_mode'] == 'pairwise':
        return get_sigmoid(preds)
    else:
        return get_softmax(preds)
