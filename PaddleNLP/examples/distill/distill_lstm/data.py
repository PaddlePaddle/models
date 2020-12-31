from functools import partial

import paddle
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall
# import paddle.nn.functional as F

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.datasets import GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

from run_bert_finetune import convert_example

TASK_CLASSES = {
    "cola": GlueCoLA,
    "sst-2": GlueSST2,
    "mrpc": GlueMRPC,
    "sts-b": GlueSTSB,
    "qqp": GlueQQP,
    "mnli": GlueMNLI,
    "qnli": GlueQNLI,
    "rte": GlueRTE,
}


def create_data_loader(task_name='sst-2',
                       batch_size=128,
                       max_seq_length=128,
                       shuffle=True):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_class = TASK_CLASSES[task_name]
    train_dataset = GlueSST2.get_datasets(['train'])
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_dataset.get_labels(),
        max_seq_length=max_seq_length)
    train_dataset = train_dataset.apply(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(),  # length
        Stack(dtype="int64" if train_dataset.get_labels() else "float32")  # label
    ): [data for i, data in enumerate(fn(samples))]  # if i != 2]

    # Create train loader
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    # Create dev loader
    if task_name == "mnli":
        dev_dataset_matched, dev_dataset_mismatched = dataset_class.get_datasets(
            ["dev_matched", "dev_mismatched"])
        dev_dataset_matched = dev_dataset_matched.apply(trans_func, lazy=True)
        dev_dataset_mismatched = dev_dataset_mismatched.apply(
            trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_dataset_matched, batch_size=batch_size, shuffle=False)
        dev_data_loader_matched = DataLoader(
            dataset=dev_dataset_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_dataset_mismatched, batch_size=batch_size, shuffle=False)
        dev_data_loader_mismatched = DataLoader(
            dataset=dev_dataset_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        return train_data_loader, dev_data_loader_matched, dev_data_loader_mismatched

    else:
        dev_dataset = dataset_class.get_datasets(["dev"])
        dev_dataset = dev_dataset.apply(trans_func, lazy=True)
        dev_batch_sampler = paddle.io.BatchSampler(
            dev_dataset, batch_size=batch_size, shuffle=False)
        dev_data_loader = DataLoader(
            dataset=dev_dataset,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        return train_data_loader, dev_data_loader


def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, _, seq_len, labels = batch
        logits = model(input_ids, seq_len)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (
                loss.numpy(),
                res[0],
                res[1],
                res[2],
                res[3],
                res[4], ),
            end='')
    elif isinstance(metric, Mcc):
        print("eval loss: %f, mcc: %s, " % (loss.numpy(), res[0]), end='')
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.numpy(), res[0], res[1], res[2]),
            end='')
    else:
        print("eval loss: %f, acc: %s, " % (loss.numpy(), res), end='')
    model.train()
