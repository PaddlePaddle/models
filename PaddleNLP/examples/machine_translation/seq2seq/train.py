# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from args import parse_args
import numpy as np

import paddle
import paddle.nn as nn
from paddle.metric import Metric

from seq2seq_attn import Seq2SeqAttnModel, CrossEntropyCriterion
from data import create_data_loader


class TrainCallback(paddle.callbacks.ProgBarLogger):
    def __init__(self, ppl, log_freq, verbose=2):
        super(TrainCallback, self).__init__(log_freq, verbose)
        self.ppl = ppl

    def on_train_begin(self, logs=None):
        super(TrainCallback, self).on_train_begin(logs)
        self.train_metrics = ["loss", "ppl"]

    def on_epoch_begin(self, epoch=None, logs=None):
        super(TrainCallback, self).on_epoch_begin(epoch, logs)
        self.ppl.reset()

    def on_train_batch_end(self, step, logs=None):
        logs["ppl"] = self.ppl.cal_acc_ppl(logs["loss"][0], logs["batch_size"])
        if step > 0 and step % self.ppl.reset_freq == 0:
            self.ppl.reset()
        super(TrainCallback, self).on_train_batch_end(step, logs)

    def on_eval_begin(self, logs=None):
        super(TrainCallback, self).on_eval_begin(logs)
        self.eval_metrics = ["ppl"]
        self.ppl.reset()

    def on_eval_batch_end(self, step, logs=None):
        logs["ppl"] = self.ppl.cal_acc_ppl(logs["loss"][0], logs["batch_size"])
        super(TrainCallback, self).on_eval_batch_end(step, logs)


class Perplexity(Metric):
    def __init__(self, reset_freq=100, name=None):
        super(Perplexity, self).__init__()
        self._name = name or "Perplexity"
        self.reset_freq = reset_freq
        self.reset()

    def compute(self, pred, seq_length, label):
        word_num = paddle.sum(seq_length)
        return word_num

    def update(self, word_num):
        self.word_count += word_num
        return word_num

    def reset(self):
        self.total_loss = 0
        self.word_count = 0

    def accumulate(self):
        return self.word_count

    def name(self):
        return self._name

    def cal_acc_ppl(self, batch_loss, batch_size):
        self.total_loss += batch_loss * batch_size
        ppl = math.exp(self.total_loss / self.word_count)
        return ppl


def do_train(args):
    device = paddle.set_device("gpu" if args.use_gpu else "cpu")

    # Define dataloader
    (train_loader, eval_loader), eos_id = create_data_loader(args, device)

    model = paddle.Model(
        Seq2SeqAttnModel(args.src_vocab_size, args.trg_vocab_size,
                         args.hidden_size, args.hidden_size, args.num_layers,
                         args.dropout, eos_id))

    grad_clip = nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        grad_clip=grad_clip)

    ppl_metric = Perplexity(reset_freq=args.log_freq)
    model.prepare(optimizer, CrossEntropyCriterion(), ppl_metric)

    print(args)
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    model.fit(train_data=train_loader,
              eval_data=eval_loader,
              epochs=args.max_epoch,
              eval_freq=1,
              save_freq=1,
              save_dir=args.model_path,
              log_freq=args.log_freq,
              callbacks=[TrainCallback(ppl_metric, args.log_freq)])


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
