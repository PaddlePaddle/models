#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

try:
    from configparser import ConfigParser
except:
    from ConfigParser import ConfigParser

from utils import AttrDict

import logging
logger = logging.getLogger(__name__)

CONFIG_SECS = [
    'train',
    'valid',
    'test',
    'infer',
]


def parse_config(cfg_file):
    parser = ConfigParser()
    cfg = AttrDict()
    parser.read(cfg_file)
    for sec in parser.sections():
        sec_dict = AttrDict()
        for k, v in parser.items(sec):
            try:
                v = eval(v)
            except:
                pass
            setattr(sec_dict, k, v)
        setattr(cfg, sec.upper(), sec_dict)

    return cfg


def merge_configs(cfg, sec, args_dict):
    assert sec in CONFIG_SECS, "invalid config section {}".format(sec)
    sec_dict = getattr(cfg, sec.upper())
    for k, v in args_dict.items():
        if v is None:
            continue
        try:
            if hasattr(sec_dict, k):
                setattr(sec_dict, k, v)
        except:
            pass
    return cfg


def print_configs(cfg, mode):
    logger.info("---------------- {:>5} Arguments ----------------".format(mode))
    for sec, sec_items in cfg.items():
        logger.info("{}:".format(sec))
        for k, v in sec_items.items():
            logger.info("    {}:{}".format(k, v))
    logger.info("-------------------------------------------------")
