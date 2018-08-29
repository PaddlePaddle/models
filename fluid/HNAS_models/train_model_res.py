# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import app
from absl import flags

import nn_paddle as nn
from build import resnet_base

FLAGS = flags.FLAGS
flags.DEFINE_string("tokdir", "tokens/", "token directory")
flags.DEFINE_integer("model", 0, "model")

mid = [17754, 15113, 15613]


def main(_):
    f = os.path.join(FLAGS.tokdir, str(mid[FLAGS.model]) + ".pkl")
    tokens = pickle.load(open(f, "rb"))

    model = nn.Model(resnet_base.net, tokens)
    model.run()


if __name__ == "__main__":
    app.run(main)
