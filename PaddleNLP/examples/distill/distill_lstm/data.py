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


def apply_data_augmentation_for_text(task_name,
                                     train_dataset,
                                     n_iter=20,
                                     p_mask=0.1,
                                     p_ng=0.25,
                                     ngram_range=(2, 6)):
    """
    Data augmentation for text, not for tokenized text list. Maybe its'
    a wrong method, because '[MASK]' in text will not be transformed to
    '[UNK]' or '[MASK]' in English mode, but '[ mask ]'.
    """
    used_texts = [data[0] for data in train_dataset]
    if task_name == 'qqp' or task_name == 'mnli':
        used_texts += [data[1] for data in train_dataset]
    used_texts = set(used_texts)
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
            if task_name == 'qqp' or task_name == 'mnli':
                # masking
                words = [
                    "[MASK]" if np.random.rand() < p_mask else word
                    for word in data[1].split()
                ]
                # n-gram sampling
                if np.random.rand() < p_ng:
                    ngram_len = np.random.randint(ngram_range[0],
                                                  ngram_range[1] + 1)
                    ngram_len = min(ngram_len, len(words))
                    start = np.random.randint(0, len(words) - ngram_len + 1)
                    words = words[start:start + ngram_len]
                new_text_2 = " ".join(words)
                if new_text not in used_texts or new_text_2 not in used_texts:
                    new_data.append([new_text, new_text_2, data[2]])
                    used_texts.add(new_text)
                    used_texts.add(new_text_2)
            else:
                if new_text not in used_texts:
                    new_data.append([new_text, data[1]])
                    used_texts.add(new_text)

    train_dataset.data.data = new_data
    print("Data augmentation is applied.")
    return train_dataset


def apply_data_augmentation(task_name,
                            train_dataset,
                            tokenizer,
                            n_iter=20,
                            p_mask=0.1,
                            p_ng=0.25,
                            ngram_range=(2, 6)):
    """
    Data Augmentation after texts are tokenized. 
    For example:
    train_ds[0] = [['hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units'], '0']
    """
    used_texts = []
    if task_name == 'qqp' or task_name == 'mnli':
        tokenized_text = [[tokenizer(data[0]), tokenizer(data[1]), data[2]]
                          for data in train_dataset]
        used_texts += [" ".join(token_list[0]) for token_list in tokenized_text]
        used_texts += [" ".join(token_list[1]) for token_list in tokenized_text]
    else:
        tokenized_text = [[tokenizer(data[0]), data[1]]
                          for data in train_dataset]
        used_texts += [" ".join(token_list[0]) for token_list in tokenized_text]

    used_texts = set(used_texts)
    new_data = []

    for idx in range(len(tokenized_text)):
        new_data.append(tokenized_text[idx])
        for _ in range(n_iter):
            # masking
            words = [
                tokenizer.unk_token if np.random.rand() < p_mask and
                not word.startswith('##') else word
                for word in tokenized_text[idx][0]
            ]
            # n-gram sampling
            if np.random.rand() < p_ng:
                ngram_len = np.random.randint(ngram_range[0],
                                              ngram_range[1] + 1)
                ngram_len = min(ngram_len, len(words))
                start = np.random.randint(0, len(words) - ngram_len + 1)
                words = words[start:start + ngram_len]
            new_text = " ".join(words)
            if task_name == 'qqp' or task_name == 'mnli':
                # masking
                words_2 = [
                    tokenizer.unk_token if np.random.rand() < p_mask and
                    not word.startswith('##') else word
                    for word in tokenized_text[idx][1]
                ]
                # n-gram sampling
                if np.random.rand() < p_ng:
                    ngram_len = np.random.randint(ngram_range[0],
                                                  ngram_range[1] + 1)
                    ngram_len = min(ngram_len, len(words_2))
                    start = np.random.randint(0, len(words_2) - ngram_len + 1)
                    words_2 = words_2[start:start + ngram_len]
                new_text_2 = " ".join(words)
                if new_text not in used_texts or new_text_2 not in used_texts:
                    new_data.append([words, words_2, tokenized_text[idx][2]])
                    used_texts.add(new_text)
                    used_texts.add(new_text_2)
            else:
                if new_text not in used_texts:
                    new_data.append([words, tokenized_text[idx][1]])
                    used_texts.add(new_text)

    train_dataset.data.data = new_data
    print("Data augmentation is applied.")

    return train_dataset


def apply_data_augmentation_for_cn(train_dataset,
                                   n_iter=20,
                                   p_mask=0.1,
                                   p_ng=0.25,
                                   ngram_range=(2, 6)):
    """Only Chinese dataset could be augmentated on raw data.
    Because bert-base-uncased and jieba could both convert 'UNK' to '[UNK]'.
    """
    used_texts = [data[0] for data in train_dataset]
    used_texts = set(used_texts)
    new_data = []
    for data in train_dataset:
        new_data.append(data)
        for _ in range(n_iter):
            # masking
            words = [
                "UNK" if np.random.rand() < p_mask else word
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
                used_texts.add(new_text)

    train_dataset.data.data = new_data
    print("Data augmentation is applied.")
    return train_dataset


def create_data_loader_for_small_model(
        task_name='sst-2',
        language='en',
        vocab_path='/root/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt',
        batch_size=64,
        max_seq_length=128,
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
        convert_small_example,
        vocab=vocab,
        language=language,
        max_seq_length=max_seq_length,
        is_test=False)

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
        data_augmentation=False,
        n_iter=20):
    """
    Returns batch data for bert and small model.
    Bert and small model have difference input representations.
    """
    print(task_name, language, model_name, vocab_path, batch_size,
          max_seq_length, use_gpu, shuffle, data_augmentation)
    dataset_class = TASK_CLASSES[task_name]
    train_ds, dev_ds = dataset_class.get_datasets(['train', 'dev'])
    tokenizer = BertTokenizer.from_pretrained(model_name)
    if language == 'cn':
        train_ds = ChnSentiCorpDataAug.get_datasets("train")
        print("ChnSentiCorp augmentation dataset has been loaded.")
        train_ds = apply_data_augmentation_for_cn(
            train_ds, n_iter=n_iter) if data_augmentation else train_ds
    else:
        train_ds = apply_data_augmentation(
            task_name, train_ds, tokenizer,
            n_iter=n_iter) if data_augmentation else train_ds

    if language == 'en':
        vocab = tokenizer
        pad_val = tokenizer.pad_token_id
    else:
        vocab = Vocab.load_vocabulary(
            vocab_path,
            unk_token='[UNK]',
            pad_token='[PAD]',
            bos_token=None,
            eos_token=None, )
        pad_val = vocab['[PAD]']

    trans_fn = partial(
        convert_two_example,
        task_name=task_name,
        tokenizer=tokenizer,
        label_list=train_ds.get_labels(),
        max_seq_length=max_seq_length,
        vocab=vocab,
        language=language)

    trans_fn_dev = partial(
        convert_two_example,
        task_name=task_name,
        tokenizer=tokenizer,
        label_list=train_ds.get_labels(),
        max_seq_length=max_seq_length,
        vocab=vocab,
        is_tokenized=False,
        language=language)

    if task_name == 'qqp' or task_name == 'mnli':
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # bert input
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # bert segment
            Pad(axis=0, pad_val=pad_val),  # small input_ids
            Stack(dtype="int64"),  # small seq len
            Pad(axis=0, pad_val=pad_val),  # small input_ids
            Stack(dtype="int64"),  # small seq len
            Stack(dtype="int64")  # small label
        ): [data for data in fn(samples)]
    else:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # bert input
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # bert segment
            Pad(axis=0, pad_val=pad_val),  # small input_ids
            Stack(dtype="int64"),  # small seq len
            Stack(dtype="int64")  # small label
        ): [data for data in fn(samples)]

    train_ds = train_ds.apply(trans_fn, lazy=True)
    dev_ds = dev_ds.apply(trans_fn_dev, lazy=True)

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
        self.add_raw_chn()

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

    def add_raw_chn(self):
        train_ds = ChnSentiCorp.get_datasets(["train"])
        for i in range(len(train_ds)):
            self.data.append(train_ds[i])

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
