from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import app
from absl import flags

import nn_paddle as nn
from build import inception_base

FLAGS = flags.FLAGS

flags.DEFINE_string("tokdir", "tokens/", "token directory")

chunk_size = 5000


def main(_):
    with open(os.path.join(FLAGS.tokdir, "inception_like.pkl"), "rb") as f:
        action = pickle.load(f)

    model = nn.Model(inception_base.net, action)
    model.run()


if __name__ == "__main__":
    app.run(main)
