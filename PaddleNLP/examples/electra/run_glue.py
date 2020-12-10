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

import argparse
import logging
import os
import sys
import hashlib
import random
import time
import math
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.datasets import GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import ElectraForSequenceClassification, ElectraTokenizer

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class AccuracyAndF1(Metric):
    """
    Encapsulates Accuracy, Precision, Recall and F1 metric logic.
    """

    def __init__(self,
                 topk=(1, ),
                 pos_label=1,
                 name='acc_and_f1',
                 *args,
                 **kwargs):
        super(AccuracyAndF1, self).__init__(*args, **kwargs)
        self.topk = topk
        self.pos_label = pos_label
        self._name = name
        self.acc = Accuracy(self.topk, *args, **kwargs)
        self.precision = Precision(*args, **kwargs)
        self.recall = Recall(*args, **kwargs)
        self.reset()

    def compute(self, pred, label, *args):
        self.label = label
        self.preds_pos = paddle.nn.functional.softmax(pred)[:, self.pos_label]
        return self.acc.compute(pred, label)

    def update(self, correct, *args):
        self.acc.update(correct)
        self.precision.update(self.preds_pos, self.label)
        self.recall.update(self.preds_pos, self.label)

    def accumulate(self):
        acc = self.acc.accumulate()
        precision = self.precision.accumulate()
        recall = self.recall.accumulate()
        if precision == 0.0 or recall == 0.0:
            f1 = 0.0
        else:
            # 1/f1 = 1/2 * (1/precision + 1/recall)
            f1 = (2 * precision * recall) / (precision + recall)
        return (
            acc,
            precision,
            recall,
            f1,
            (acc + f1) / 2, )

    def reset(self):
        self.acc.reset()
        self.precision.reset()
        self.recall.reset()
        self.label = None
        self.preds_pos = None

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name


class Mcc(Metric):
    """
    Matthews correlation coefficient
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient.
    """

    def __init__(self, name='mcc', *args, **kwargs):
        super(Mcc, self).__init__(*args, **kwargs)
        self._name = name
        self.tp = 0  # true positive
        self.fp = 0  # false positive
        self.tn = 0  # true negative
        self.fn = 0  # false negative

    def compute(self, pred, label, *args):
        preds = paddle.argsort(pred, descending=True)[:, :1]
        return (preds, label)

    def update(self, preds_and_labels):
        preds = preds_and_labels[0]
        preds = preds.numpy()
        labels = preds_and_labels[1]
        labels = labels.numpy().reshape(-1, 1)
        sample_num = labels.shape[0]
        for i in range(sample_num):
            pred = preds[i]
            label = labels[i]
            if pred == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if pred == label:
                    self.tn += 1
                else:
                    self.fn += 1

    def accumulate(self):
        if self.tp == 0 or self.fp == 0 or self.tn == 0 or self.fn == 0:
            mcc = 0.0
        else:
            # mcc = (tp*tn-fp*fn)/ sqrt(tp+fp)(tp+fn)(tn+fp)(tn+fn))
            mcc = (self.tp * self.tn - self.fp * self.fn) / math.sqrt(
                (self.tp + self.fp) * (self.tp + self.fn) *
                (self.tn + self.fp) * (self.tn + self.fn))
        return (mcc, )

    def reset(self):
        self.tp = 0  # true positive
        self.fp = 0  # false positive
        self.tn = 0  # true negative
        self.fn = 0  # false negative

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name


class PearsonAndSpearman(Metric):
    """
    Pearson correlation coefficient
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    Spearman's rank correlation coefficient
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient.
    """

    def __init__(self, name='mcc', *args, **kwargs):
        super(PearsonAndSpearman, self).__init__(*args, **kwargs)
        self._name = name
        self.preds = []
        self.labels = []

    def update(self, preds_and_labels):
        preds = preds_and_labels[0]
        preds = np.squeeze(preds.numpy().reshape(-1, 1)).tolist()
        labels = preds_and_labels[1]
        labels = np.squeeze(labels.numpy().reshape(-1, 1)).tolist()
        self.preds.append(preds)
        self.labels.append(labels)

    def accumulate(self):
        preds = [item for sublist in self.preds for item in sublist]
        labels = [item for sublist in self.labels for item in sublist]
        #import pdb; pdb.set_trace()
        pearson = self.pearson(preds, labels)
        spearman = self.spearman(preds, labels)
        return (
            pearson,
            spearman,
            (pearson + spearman) / 2, )

    def pearson(self, preds, labels):
        n = len(preds)
        #simple sums
        sum1 = sum(float(preds[i]) for i in range(n))
        sum2 = sum(float(labels[i]) for i in range(n))
        #sum up the squares
        sum1_pow = sum([pow(v, 2.0) for v in preds])
        sum2_pow = sum([pow(v, 2.0) for v in labels])
        #sum up the products
        p_sum = sum([preds[i] * labels[i] for i in range(n)])

        numerator = p_sum - (sum1 * sum2 / n)
        denominator = math.sqrt(
            (sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def spearman(self, preds, labels):
        preds_rank = self.get_rank(preds)
        labels_rank = self.get_rank(labels)

        total = 0
        n = len(preds)
        for i in range(n):
            total += pow((preds_rank[i] - labels_rank[i]), 2)
        spearman = 1 - float(6 * total) / (n * (pow(n, 2) - 1))
        return spearman

    def get_rank(self, raw_list):
        x = np.array(raw_list)
        r_x = np.empty(x.shape, dtype=int)
        y = np.argsort(-x)
        for i, k in enumerate(y):
            r_x[k] = i + 1
        return r_x

    def reset(self):
        self.preds = []
        self.labels = []

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name


TASK_CLASSES = {
    "cola": (GlueCoLA, Mcc),
    "sst-2": (GlueSST2, Accuracy),
    "mrpc": (GlueMRPC, AccuracyAndF1),
    "sts-b": (GlueSTSB, PearsonAndSpearman),
    "qqp": (GlueQQP, AccuracyAndF1),
    "mnli": (GlueMNLI, Accuracy),
    "qnli": (GlueQNLI, Accuracy),
    "rte": (GlueRTE, Accuracy),
}

MODEL_CLASSES = {
    "electra": (ElectraForSequenceClassification, ElectraTokenizer),
}


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids=input_ids, token_type_ids=segment_ids)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    acc = metric.accumulate()
    print("eval loss: %f, acc: %s, " % (loss.numpy(), acc), end='')
    model.train()


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=128,
                    is_test=False):
    """convert a glue example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        if len(seqs) == 1:  # single sentence
            # Account for [CLS] and [SEP] with "- 2"
            seqs[0] = seqs[0][0:(max_seq_length - 2)]
        else:  # Sentence pair
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = seqs
            max_seq_length -= 3
            while True:  # Truncate with longest_first strategy
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_seq_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        return seqs

    def _concat_seqs(seqs, separators, seq_mask=0, separator_mask=1):
        concat = sum((seq + sep for sep, seq in zip(separators, seqs)), [])
        segment_ids = sum(
            ([i] * (len(seq) + len(sep))
             for i, (sep, seq) in enumerate(zip(separators, seqs))), [])
        if isinstance(seq_mask, int):
            seq_mask = [[seq_mask] * len(seq) for seq in seqs]
        if isinstance(separator_mask, int):
            separator_mask = [[separator_mask] * len(sep) for sep in separators]
        p_mask = sum((s_mask + mask
                      for sep, seq, s_mask, mask in zip(
                          separators, seqs, seq_mask, separator_mask)), [])
        return concat, segment_ids, p_mask

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example[-1]
        example = example[:-1]
        # Create label maps if classification task
        if label_list:
            label_map = {}
            for (i, l) in enumerate(label_list):
                label_map[l] = i
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)

    # Tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # Truncate to the truncate_length,
    tokens_trun = _truncate_seqs(tokens_raw, max_seq_length)
    # Concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # Convert the token to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    valid_length = len(input_ids)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    # input_mask = [1] * len(input_ids)
    if not is_test:
        return input_ids, segment_ids, valid_length, label
    else:
        return input_ids, segment_ids, valid_length


def do_train(args):
    paddle.enable_static() if not args.eager_run else None
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    dataset_class, metric_class = TASK_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_dataset = dataset_class.get_datasets(["train"])
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_dataset.get_labels(),
        max_seq_length=args.max_seq_length)
    train_dataset = train_dataset.apply(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(),  # length
        Stack(dtype="int64" if train_dataset.get_labels() else "float32")  # label
    ): [data for i, data in enumerate(fn(samples)) if i != 2]
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)
    if args.task_name == "mnli":
        dev_dataset_matched, dev_dataset_mismatched = dataset_class.get_datasets(
            ["dev_matched", "dev_mismatched"])
        dev_dataset_matched = dev_dataset_matched.apply(trans_func, lazy=True)
        dev_dataset_mismatched = dev_dataset_mismatched.apply(
            trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_dataset_matched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_matched = DataLoader(
            dataset=dev_dataset_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_dataset_mismatched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_mismatched = DataLoader(
            dataset=dev_dataset_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
    else:
        dev_dataset = dataset_class.get_datasets(["dev"])
        dev_dataset = dev_dataset.apply(trans_func, lazy=True)
        dev_batch_sampler = paddle.io.BatchSampler(
            dev_dataset, batch_size=args.batch_size, shuffle=False)
        dev_data_loader = DataLoader(
            dataset=dev_dataset,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    num_labels = 1 if train_dataset.get_labels() == None else len(
        train_dataset.get_labels())
    model = model_class.from_pretrained(
        args.model_name_or_path, num_labels=num_labels)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)
    warmup_steps = int(math.floor(num_training_steps * args.warmup_proportion))
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        args.learning_rate,
        lambda current_step, num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps : float(
            current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps else max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps))))

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm", "LayerNorm"])
        ])

    loss_fct = paddle.nn.loss.CrossEntropyLoss() if train_dataset.get_labels(
    ) else paddle.nn.loss.MSELoss()

    metric = metric_class()

    ### TODO: use hapi
    # trainer = paddle.hapi.Model(model)
    # trainer.prepare(optimizer, loss_fct, paddle.metric.Accuracy())
    # trainer.fit(train_data_loader,
    #             dev_data_loader,
    #             log_freq=args.logging_steps,
    #             epochs=args.num_train_epochs,
    #             save_dir=args.output_dir)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, labels = batch
            logits = model(input_ids=input_ids, token_type_ids=segment_ids)
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
            if global_step % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.save_steps == 0:
                tic_eval = time.time()
                if args.task_name == "mnli":
                    evaluate(model, loss_fct, metric, dev_data_loader_matched)
                    evaluate(model, loss_fct, metric,
                             dev_data_loader_mismatched)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                else:
                    evaluate(model, loss_fct, metric, dev_data_loader)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "%s_ft_model_%d.pdparams" %
                                              (args.task_name, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)


def get_md5sum(file_path):
    md5sum = None
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            md5_obj = hashlib.md5()
            md5_obj.update(f.read())
            hash_code = md5_obj.hexdigest()
        md5sum = str(hash_code).lower()
    return md5sum


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(TASK_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default="electra",
        type=str,
        required=False,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default="./",
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default="./ft_model/",
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--eager_run", default=True, type=eval, help="Use dygraph mode.")
    parser.add_argument(
        "--n_gpu",
        default=1,
        type=int,
        help="number of gpus to use, 0 for cpu.")
    args, unparsed = parser.parse_known_args()
    print_arguments(args)
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
