#encoding=utf8
import os
import sys
import logging

import numpy as np
import paddle
import paddle.fluid as fluid

#include palm for easier nlp coding
from palm.toolkit.configure import PDConfig

from train import do_train
from predict import do_predict
from inference_model import do_save_inference_model

if __name__ == "__main__":
    LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG, format=LOG_FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()

    if args.do_train:
        do_train(args)

    if args.do_predict:
        do_predict(args)

    if args.do_save_inference_model:
        do_save_inference_model(args)