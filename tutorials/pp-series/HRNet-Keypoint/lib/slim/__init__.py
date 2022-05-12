from . import quant

from .quant import *

import yaml
from lib.utils.workspace import load_config, create
from lib.utils.checkpoint import load_pretrain_weight


def build_slim_model(cfg, mode='train'):
    assert cfg.slim == 'QAT', 'Only QAT is supported now'
    model = create(cfg.architecture)
    if mode == 'train':
        load_pretrain_weight(model, cfg.pretrain_weights)
    slim = create(cfg.slim)
    cfg['slim_type'] = cfg.slim
    # TODO: fix quant export model in framework.
    if mode == 'test' and cfg.slim == 'QAT':
        slim.quant_config['activation_preprocess_type'] = None
    cfg['model'] = slim(model)
    cfg['slim'] = slim
    if mode != 'train':
        load_pretrain_weight(cfg['model'], cfg.weights)

    return cfg
