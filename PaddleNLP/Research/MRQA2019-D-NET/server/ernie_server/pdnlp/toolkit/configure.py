#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import six
import logging
import json

logging_only_message = "%(message)s"
logging_details = "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"

class JsonConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except:
            raise IOError("Error in parsing bert model config file '%s'" %
                config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)

class ArgConfig(object):
    
    def __init__(self):
        parser = argparse.ArgumentParser()

        train_g = ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg("epoch",             int,    3,      "Number of epoches for fine-tuning.")
        train_g.add_arg("learning_rate",     float,  5e-5,   "Learning rate used to train with warmup.")
        train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                        "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
        train_g.add_arg("weight_decay",      float,  0.01,   "Weight decay rate for L2 regularizer.")
        train_g.add_arg("warmup_proportion", float,  0.1,
                        "Proportion of training steps to perform linear learning rate warmup for.")
        train_g.add_arg("save_steps",        int,    1000,   "The steps interval to save checkpoints.")
        train_g.add_arg("use_fp16",          bool,   False,  "Whether to use fp16 mixed precision training.")
        train_g.add_arg("loss_scaling",      float,  1.0,
                        "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
        train_g.add_arg("pred_dir",   str,    None,   "Path to save the prediction results")

        log_g = ArgumentGroup(parser, "logging", "logging related.")
        log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")
        log_g.add_arg("verbose",             bool,   False, "Whether to output verbose log.")

        run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
        run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
        run_type_g.add_arg("use_fast_executor",            bool,   False, "If set, use fast parallel executor (in experiment).")
        run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,     "Ihe iteration intervals to clean up temporary variables.")
        run_type_g.add_arg("do_train",                     bool,   True,  "Whether to perform training.")
        run_type_g.add_arg("do_predict",                   bool,   True,  "Whether to perform prediction.")

        custom_g = ArgumentGroup(parser, "customize", "customized options.")

        self.custom_g = custom_g

        self.parser = parser

    def add_arg(self, name, dtype, default, descrip):
        self.custom_g.add_arg(name, dtype, default, descrip)

    def build_conf(self):
        return self.parser.parse_args()


def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


def print_arguments(args, log = None):
    if not log:
        print('-----------  Configuration Arguments -----------')
        for arg, value in sorted(six.iteritems(vars(args))):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')
    else:
        log.info('-----------  Configuration Arguments -----------')
        for arg, value in sorted(six.iteritems(vars(args))):
            log.info('%s: %s' % (arg, value))
        log.info('------------------------------------------------')


if __name__ == "__main__":

    args = ArgConfig()
    args = args.build_conf()

    # using print()
    print_arguments(args)

    logging.basicConfig(
        level=logging.INFO,
        format=logging_details,
        datefmt='%Y-%m-%d %H:%M:%S')

    # using logging
    print_arguments(args, logging)

    json_conf = JsonConfig("../../data/pretrained_models/uncased_L-12_H-768_A-12/bert_config.json")
    json_conf.print_config()



