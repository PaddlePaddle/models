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

from args import parse_args

from data import create_train_loader
from model import Seq2SeqAttnModel, CrossEntropyCriterion

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Perplexity(paddle.metric.Metric):
    def __init__(self, name='Perplexity', *args, **kwargs):
        super(Perplexity, self).__init__(*args, **kwargs)
        self._name = name
        self.total_ce = 0
        self.word_count = 0

    def compute(self, pred, label, seq_mask=None):
        label = paddle.unsqueeze(label, axis=2)
        ce = F.softmax_with_cross_entropy(
            logits=pred, label=label, soft_label=False)
        ce = paddle.squeeze(ce, axis=[2])
        if seq_mask is not None:
            ce = ce * seq_mask
            word_num = paddle.sum(seq_mask)  # [0]
            return ce, word_num
        return ce

    def update(self, ce, word_num=None):
        batch_ce = np.sum(ce)
        if word_num is None:
            word_num = ce.shape[0] * ce.shape[1]
        self.total_ce += batch_ce
        self.word_count += word_num

    def reset(self):
        self.total_ce = 0
        self.word_count = 0

    def accumulate(self):
        return np.exp(self.total_ce / self.word_count)[0]

    def name(self):
        return self._name


def do_train(args):
    device = paddle.set_device("gpu" if args.use_gpu else "cpu")

    # Define dataloader
    train_loader, dev_loader, vocab_size, pad_id = create_train_loader(
        args.batch_size)

    model = paddle.Model(
        Seq2SeqAttnModel(vocab_size, args.hidden_size, args.hidden_size,
                         args.num_layers, args.dropout, pad_id))

    grad_clip = nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        grad_clip=grad_clip)

    ppl_metric = Perplexity()
    model.prepare(optimizer, CrossEntropyCriterion(), ppl_metric)

    print(args)
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    model.fit(train_data=train_loader,
              eval_data=dev_loader,
              epochs=args.max_epoch,
              eval_freq=1,
              save_freq=1,
              save_dir=args.model_path,
              log_freq=args.log_freq,
              callbacks=[paddle.callbacks.VisualDL('./log')])

    print('Start to evaluate on development dataset...')
    model.evaluate(dev_loader, log_freq=len(dev_loader))


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
