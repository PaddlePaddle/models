import io
import os
from functools import partial
import gensim
import numpy as np
import jieba

import paddle
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.data import Stack, Tuple, Pad, Vocab
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.datasets import GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE, ChnSentiCorp
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

from run_bert_finetune import convert_example
from utils import convert_small_example, convert_two_example, convert_pair_example

TASK_CLASSES = {
    "sst-2": GlueSST2,
    "qqp": GlueQQP,
    "mnli": GlueMNLI,
    "senta": ChnSentiCorp,
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


def load_embedding(
        vocab_path='/root/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt',
        emb_dim=300,
        word2vec_path='GoogleNews-vectors-negative300.bin'):
    vocab_list = []
    with io.open(vocab_path) as f:
        for line in f:
            vocab_list.append(line.strip())
    vocab_size = len(vocab_list)
    emb_np = np.zeros((vocab_size, emb_dim), dtype="float32")
    word2vec_dict = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_path, binary=True)

    for i in range(vocab_size):
        word = vocab_list[i]
        if word in word2vec_dict:
            emb_np[i] = word2vec_dict[word]
    emb_tensor = paddle.to_tensor(emb_np)
    return emb_tensor


def apply_data_augmentation(train_dataset,
                            n_iter=20,
                            p_mask=0.1,
                            p_ng=0.25,
                            ngram_range=(2, 6)):
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
                       shuffle=True):
    """Create dataloader for bert."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_class = TASK_CLASSES[task_name]
    train_ds, dev_ds = dataset_class.get_datasets(['train', 'dev'])

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.get_labels(),
        max_seq_length=max_seq_length)
    train_ds = train_ds.apply(trans_func, lazy=True)
    dev_ds = dev_ds.apply(trans_func, lazy=True)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=batch_size, shuffle=shuffle)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(),  # length
        Stack(dtype="int64" if train_ds.get_labels() else "float32")  # label
    ): [data for i, data in enumerate(fn(samples))]

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

    train_data_loader, dev_data_loader = create_dataloader(
        train_ds, dev_ds, batch_size, batchify_fn, shuffle)
    return train_data_loader, dev_data_loader


def create_data_loader_for_small_model(
        task_name='sst-2',
        language='en',
        vocab_path='/root/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt',
        batch_size=64,
        use_gpu=True,
        shuffle=True):
    """data loader for bi-lstm, not bert."""
    dataset_class = TASK_CLASSES[task_name]
    train_ds, dev_ds = dataset_class.get_datasets(['train', 'dev'])

    if language == 'cn':
        vocab = Vocab.load_vocabulary(
            vocab_path,
            unk_token='[UNK]',
            pad_token='[PAD]',
            bos_token=None,
            eos_token=None, )
        pad_val = vocab['[PAD]']

    else:
        vocab = BertTokenizer.from_pretrained('bert-base-uncased')
        pad_val = vocab.pad_token_id

    trans_fn = partial(
        convert_small_example, vocab=vocab, language=language, is_test=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_val),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    train_ds = train_ds.apply(trans_fn, lazy=True)
    dev_ds = dev_ds.apply(trans_fn, lazy=True)

    train_data_loader, dev_data_loader = create_dataloader(
        train_ds, dev_ds, batch_size, batchify_fn, shuffle)

    return train_data_loader, dev_data_loader


def create_distill_loader(
        task_name='sst-2',
        language='en',
        model_name='bert-base-uncased',
        vocab_path='/root/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt',
        batch_size=64,
        max_seq_length=128,
        use_gpu=True,
        shuffle=True,
        data_augmentation=False):
    """
    Returns batch data for bert and small model.
    Bert and small model have difference input representations.
    """
    dataset_class = TASK_CLASSES[task_name]
    train_ds, dev_ds = dataset_class.get_datasets(['train', 'dev'])
    # if language == 'cn' and data_augmentation: # Use augmentation file.
    #     train_ds = ChnSentiCorpDataAug.get_datasets("train")
    #     print("ChnSentiCorp augmentation dataset has been loaded.")
    # elif data_augmentation: # Use our augmentation function
    #     train_ds = apply_data_augmentation(
    # train_ds) if data_augmentation else train_ds
    train_ds = apply_data_augmentation(
        train_ds) if data_augmentation else train_ds

    tokenizer = BertTokenizer.from_pretrained(model_name)
    if language == 'en':
        vocab = tokenizer
    else:
        vocab = Vocab.load_vocabulary(
            vocab_path,
            unk_token='[UNK]',
            pad_token='[PAD]',
            bos_token=None,
            eos_token=None, )

    trans_fn = partial(
        convert_two_example,
        tokenizer=tokenizer,
        label_list=train_ds.get_labels(),
        max_seq_length=max_seq_length,
        vocab=vocab,
        language=language)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # bert input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # bert segment
        Pad(axis=0, pad_val=vocab['[PAD]']),  # small input_ids
        Stack(dtype="int64"),  # small seq len
        Stack(dtype="int64")  # small label
    ): [data for data in fn(samples)]

    train_ds = train_ds.apply(trans_fn, lazy=True)
    dev_ds = dev_ds.apply(trans_fn, lazy=True)

    train_data_loader, dev_data_loader = create_dataloader(
        train_ds, dev_ds, batch_size, batchify_fn, shuffle)

    return train_data_loader, dev_data_loader


def create_pair_loader_for_small_model(
        task_name='qqp',
        language='en',
        vocab_path='/root/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt',
        batch_size=64,
        max_seq_length=128,
        use_gpu=True,
        shuffle=True,
        is_test=False,
        data_augmentation=False):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_class = TASK_CLASSES[task_name]

    train_ds, dev_ds = dataset_class.get_datasets(['train', 'dev'])
    vocab = Vocab.load_vocabulary(
        vocab_path,
        unk_token='[UNK]',
        pad_token='[PAD]',
        bos_token=None,
        eos_token=None, )

    trans_func = partial(
        convert_pair_example,
        vocab=tokenizer,
        language=language,
        max_seq_length=max_seq_length,
        is_test=is_test)
    train_ds = train_ds.apply(trans_func, lazy=True)
    dev_ds = dev_ds.apply(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab['[PAD]']),  # input
        Stack(),  # length
        Pad(axis=0, pad_val=vocab['[PAD]']),  # input
        Stack(),  # length
        Stack(dtype="int64" if train_ds.get_labels() else "float32")  # label
    ): [data for i, data in enumerate(fn(samples))]

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

    train_data_loader, dev_data_loader = create_dataloader(
        train_ds, dev_ds, batch_size, batchify_fn, shuffle)
    return train_data_loader, dev_data_loader


def create_dataloader(train_ds, dev_ds, batch_size, batchify_fn, shuffle=True):
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=batch_size, shuffle=shuffle)

    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=batch_size, shuffle=False)

    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    return train_data_loader, dev_data_loader


class ChnSentiCorpDataAug(paddle.io.Dataset):
    def __init__(self,
                 mode="train",
                 path='chnsenticorp-data/train-data-augmented/'):
        self.data = self.read_raw_data(path)

    def read_raw_data(self, path):
        filelist = os.listdir(path)
        data = []
        for filename in filelist:
            with io.open(os.path.join(path, filename)) as f:
                for line in f:
                    line = line.strip()

                    sentence = self.clean_sentence("".join(
                        line.split("\t")[:-1]))
                    label = line.split("\t")[-1]
                    data.append([sentence, label])
        return data

    def clean_sentence(self, sentence):
        while " " in sentence:
            sentence = sentence.replace(" ", "")
        return sentence

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return ["0", "1"]
