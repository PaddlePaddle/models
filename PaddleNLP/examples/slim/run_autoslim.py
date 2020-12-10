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
import random
import time
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader

from paddlenlp.datasets import GlueQNLI, GlueSST2
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import BertModel, BertForSequenceClassification, BertTokenizer
from paddleslim.nas.ofa.utils import compute_neuron_head_importance, reorder_head, reorder_neuron, set_state_dict
from paddleslim.nas.ofa import OFA, DistillConfig
from paddleslim.nas.ofa.convert_super import Convert, supernet

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

TASK_CLASSES = {
    "qnli": (GlueQNLI, paddle.metric.Accuracy),  # (dataset, metric)
    "sst-2": (GlueSST2, paddle.metric.Accuracy),
}

MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer), }


def parse_args():
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
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--lambda_logit",
        default=1.0,
        type=float,
        help="lambda for logit loss.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--eager_run", type=eval, default=True, help="Use dygraph mode.")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    parser.add_argument(
        '--width_mult_list',
        nargs='+',
        type=float,
        default=[1.0, 5 / 6, 2 / 3, 0.5],
        help="width mult in compress")
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


def evaluate(model, loss_fct, metric, data_loader, width_mult=1.0):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids, attention_mask=[None, None])
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("width_mult: %f, eval loss: %f, accu: %f" %
          (width_mult, loss.numpy(), accu))
    model.train()
    return accu


def bert_forward(self,
                 input_ids,
                 token_type_ids=None,
                 position_ids=None,
                 attention_mask=[None, None]):
    if attention_mask[0] is None:
        attention_mask[0] = paddle.unsqueeze(
            (input_ids == self.pad_token_id
             ).astype(self.pooler.dense.fn.weight.dtype) * -1e9,
            axis=[1, 2])
    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids)
    encoder_outputs = self.encoder(embedding_output, attention_mask)
    sequence_output = encoder_outputs
    pooled_output = self.pooler(sequence_output)
    return sequence_output, pooled_output


BertModel.forward = bert_forward


def reorder_neuron_head(model, head_importance, neuron_importance):
    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = paddle.argsort(head_importance[layer], descending=True)
        reorder_head(model.bert.encoder.layers[layer].self_attn, idx)
        # reorder neurons
        idx = paddle.argsort(
            paddle.to_tensor(current_importance), descending=True)
        reorder_neuron(model.bert.encoder.layers[layer].linear1.fn, idx, dim=1)
        reorder_neuron(model.bert.encoder.layers[layer].linear2.fn, idx, dim=0)


@paddle.no_grad()
def adamw_step(optim, params_grads):
    scaled_params = optim._scale_parameters(params_grads)
    for p_grad_sgrad in scaled_params:
        param, grad, scaled_param = p_grad_sgrad
        with param.block.program._optimized_guard(
            [param, grad]), paddle.static.name_scope('weight decay'):
            updated_param = paddle.fluid.layers.elementwise_sub(
                x=param, y=scaled_param)
            paddle.assign(x=updated_param, output=param)
    optim._apply_optimize(
        loss=None, startup_program=None, params_grads=params_grads)


def soft_cross_entropy(inp, target):
    inp_likelihood = F.log_softmax(inp, axis=-1)
    target_prob = F.softmax(target, axis=-1)
    return -1. * paddle.mean(paddle.sum(inp_likelihood * target_prob, axis=-1))


def apply_config(model, width_mult):
    new_config = dict()

    def fix_exp(idx):
        if (idx - 3) % 6 == 0 or (idx - 5) % 6 == 0:
            return True
        return False

    for idx, (block_k, block_v) in enumerate(model.layers.items()):
        if len(block_v.keys()) != 0:
            name, name_idx = block_k.split('_'), int(block_k.split('_')[1])
            if fix_exp(name_idx) or 'emb' in block_k or idx == (
                    len(model.layers.items()) - 2):
                block_v['expand_ratio'] = 1.0
            else:
                block_v['expand_ratio'] = width_mult
        new_config[block_k] = block_v
    return new_config


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        if len(seqs) == 1:  # single sentence
            # Account for [CLS] and [SEP] with "- 2"
            seqs[0] = seqs[0][0:(max_seq_length - 2)]
        else:  # sentence pair
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = seqs
            max_seq_length -= 3
            while True:  # truncate with longest_first strategy
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
        # get the label
        label = example[-1]
        example = example[:-1]
        #create label maps if classification task
        if label_list:
            label_map = {}
            for (i, l) in enumerate(label_list):
                label_map[l] = i
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)

    # tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # truncate to the truncate_length,
    tokens_trun = _truncate_seqs(tokens_raw, max_seq_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # convert the token to ids
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

    #width_mult_list = [1.0, 0.75, 0.5, 0.25]
    args.task_name = args.task_name.lower()
    dataset_class, metric_class = TASK_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_ds, dev_ds = dataset_class.get_datasets(['train', 'dev'])

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.get_labels(),
        max_seq_length=args.max_seq_length)
    train_ds = train_ds.apply(trans_func, lazy=True)
    # train_batch_sampler = SamplerHelper(train_ds).shuffle().batch(
    #     batch_size=args.batch_size).shard()
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(),  # length
        Stack(dtype="int64" if train_ds.get_labels() else "float32")  # label
    ): [data for i, data in enumerate(fn(samples)) if i != 2]
    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_ds = dev_ds.apply(trans_func, lazy=True)
    # dev_batch_sampler = SamplerHelper(dev_ds).batch(
    #     batch_size=args.batch_size)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args.batch_size, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=len(train_ds.get_labels()))
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    origin_weights = {}
    for name, param in model.named_parameters():
        origin_weights[name] = param  #.numpy()

    sp_config = supernet(expand_ratio=args.width_mult_list)
    model = Convert(sp_config).convert(model)
    set_state_dict(model, origin_weights)

    teacher_model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=len(train_ds.get_labels()))

    teacher_model = Convert(sp_config).convert(teacher_model)
    set_state_dict(teacher_model, origin_weights)
    del origin_weights

    mapping_layers = ['bert.embeddings']
    for idx in range(model.bert.config['num_hidden_layers']):
        mapping_layers.append('bert.encoder.layers.{}'.format(idx))

    default_distill_config = {
        'lambda_distill': 0.1,
        'teacher_model': teacher_model,
        'mapping_layers': mapping_layers,
    }
    distill_config = DistillConfig(**default_distill_config)
    ofa_model = OFA(model,
                    distill_config=distill_config,
                    elastic_order=['width'])

    head_importance, neuron_importance = compute_neuron_head_importance(
        args.task_name,
        ofa_model.model,
        dev_data_loader,
        num_layers=model.bert.config['num_hidden_layers'],
        num_heads=model.bert.config['num_attention_heads'])
    reorder_neuron_head(ofa_model.model, head_importance, neuron_importance)

    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        args.learning_rate,
        lambda current_step, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps if args.max_steps > 0 else
        (len(train_data_loader) * args.num_train_epochs): float(
            current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps else max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps))))

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=ofa_model.model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in ofa_model.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    loss_fct = paddle.nn.loss.CrossEntropyLoss() if train_ds.get_labels(
    ) else paddle.nn.loss.MSELoss()

    metric = metric_class()

    global_step = 0
    tic_train = time.time()
    best_acc = [-1.0, -1.0, -1.0]
    for epoch in range(args.num_train_epochs):
        ofa_model.set_epoch(epoch)
        ofa_model.set_task('width')

        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, labels = batch

            accumulate_gradients = dict()
            for param in optimizer._parameter_list:
                accumulate_gradients[param.name] = 0.0

            for width_mult in args.width_mult_list:
                net_config = apply_config(ofa_model, width_mult)
                ofa_model.set_net_config(net_config)
                logits, teacher_logits = ofa_model(
                    input_ids, segment_ids, attention_mask=[None, None])
                rep_loss = ofa_model.calc_distill_loss()
                logit_loss = soft_cross_entropy(logits, teacher_logits.detach())
                loss = rep_loss + args.lambda_logit * logit_loss
                loss.backward()
                param_grads = optimizer.backward(loss)
                for param in optimizer._parameter_list:
                    accumulate_gradients[param.name] += param.gradient()

            for k, v in param_grads:
                assert k.name in accumulate_gradients.keys(
                ), "{} not in accumulate_gradients".format(k.name)
                v.set_value(accumulate_gradients[k.name])
            adamw_step(optimizer, params_grads=param_grads)
            lr_scheduler.step()
            ofa_model.model.clear_gradients()

            if global_step % args.logging_steps == 0:
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss,
                           args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0:
                saved = False
                evaluate(
                    teacher_model,
                    loss_fct,
                    metric,
                    dev_data_loader,
                    width_mult=100)
                for idx, width_mult in enumerate(args.width_mult_list):
                    net_config = apply_config(ofa_model, width_mult)
                    ofa_model.set_net_config(net_config)
                    acc = evaluate(ofa_model, loss_fct, metric, dev_data_loader,
                                   width_mult)

                    if acc > best_acc[idx] and saved == False:
                        saved = True
                        best_acc[idx] = acc
                        if (not args.n_gpu > 1
                            ) or paddle.distributed.get_rank() == 0:
                            output_dir = os.path.join(args.output_dir,
                                                      "model_%d" % global_step)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            # need better way to get inner model of DataParallel
                            model_to_save = ofa_model.model._layers if isinstance(
                                ofa_model.model,
                                paddle.DataParallel) else ofa_model.model
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
