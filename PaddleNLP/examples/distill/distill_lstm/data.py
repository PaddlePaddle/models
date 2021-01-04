from functools import partial
import numpy as np

import paddle
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.data import Stack, Tuple, Pad
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


def apply_data_augmentation(train_dataset,
                            n_iter=20,
                            p_mask=0.1,
                            p_ng=0.25,
                            ngram_range=(1, 5)):
    used_texts = [data[0] for data in train_dataset]
    new_data = []
    for data in train_dataset:
        new_data.append(data)
        for _ in range(n_iter):
            # masking
            words = [
                "[MASK]" if np.random.rand() < p_mask else word
                for word in data[0].split()
            ]
            # n-gram sampling
            if np.random.rand() < p_ng:
                ngram_len = np.random.randint(ngram_range[0],
                                              ngram_range[1] + 1)
                ngram_len = min(ngram_len, len(words))
                start = np.random.randint(0, len(words) - ngram_len + 1)
                words = words[start:start + ngram_len]
            new_text = " ".join(words)
            if new_text not in used_texts:
                new_data.append([new_text, data[1]])

    train_dataset.data.data = new_data
    print("Data augmentation is applied.")
    return train_dataset


def create_data_loader(task_name='sst-2',
                       batch_size=128,
                       max_seq_length=128,
                       shuffle=True,
                       data_augmentation=False):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_class = TASK_CLASSES[task_name]
    train_dataset = GlueSST2.get_datasets(['train'])
    if data_augmentation:
        train_dataset = apply_data_augmentation(train_dataset)
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
