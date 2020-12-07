# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from functools import partial
import argparse
import os
import random
import time

import jieba
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as ppnlp

from config import RunConfig
from data import ChnSentiCorp, convert_tokens_to_ids, load_vocab
from model import BoWModel, LSTMModel, GRUModel, RNNModel, BiLSTMAttentionModel, TextCNNModel
from model import SelfAttention, SelfInteractiveAttention

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=3, help="Number of epoches for training.")
parser.add_argument('--use_gpu', type=eval, default=False, help="Whether use GPU for training, input should be True or False")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='chekpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./word_dict.txt", help="The directory to dataset.")
parser.add_argument('--network_name', type=str, default="bilstm_attn", help="Which network you would like to choose bow, lstm, bilstm, gru, bigru, rnn, birnn, bilstm_attn and textcnn?")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
args = parser.parse_args()
# yapf: enable


def pad_texts_to_max_seq_len(texts, max_seq_len, pad_token_id=0):
    """
    Padded the texts to the max sequence length if the length of text is lower than it.
    Unless it truncates the text.

    Args:
        texts(obj:`list`): Texts which contrains a sequence of word ids.
        max_seq_len(obj:`int`): Max sequence length.
        pad_token_id(obj:`int`, optinal, defaults to 0) : The pad token index.
    """
    for index, text in enumerate(texts):
        seq_len = len(text)
        if seq_len < max_seq_len:
            padded_tokens = [pad_token_id for _ in range(max_seq_len - seq_len)]
            new_text = text + padded_tokens
            texts[index] = new_text
        elif seq_len > max_seq_len:
            new_text = text[:max_seq_len]
            texts[index] = new_text


def generate_batch(batch, pad_token_id=0, return_label=True):
    """
    Generates a batch whose text will be padded to the max sequence length in the batch.

    Args:
        batch(obj:`List[Example]`) : One batch, which contains texts, labels and the true sequence lengths.
        pad_token_id(obj:`int`, optinal, defaults to 0) : The pad token index.

    Returns:
        batch(:obj:`Tuple[list]`): The batch data which contains texts, seq_lens and labels.
    """
    seq_lens = [entry[1] for entry in batch]

    batch_max_seq_len = max(seq_lens)
    texts = [entry[0] for entry in batch]
    pad_texts_to_max_seq_len(texts, batch_max_seq_len, pad_token_id)

    if return_label:
        labels = [[entry[-1]] for entry in batch]
        return texts, seq_lens, labels
    else:
        return texts, seq_lens


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      use_gpu=False,
                      pad_token_id=0):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        use_gpu(obj:`bool`, optional, defaults to obj:`False`): Whether to use gpu to run.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.apply(trans_fn, lazy=True)

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=True)
        dataloader = paddle.io.DataLoader(
            dataset,
            batch_sampler=sampler,
            return_list=True,
            collate_fn=lambda batch: generate_batch(batch,
                                                    pad_token_id=pad_token_id))
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader = paddle.io.DataLoader(
            dataset,
            batch_sampler=sampler,
            return_list=True,
            collate_fn=lambda batch: generate_batch(batch,
                                                    pad_token_id=pad_token_id))
    return dataloader


def create_model(vocab_size, num_labels, network_name='bilstm', pad_token_id=0):
    """
    Creats model which uses to text classification. It should be BoW, LSTM/BiLSTM, GRU/BiGRU.

    Args:
        vocab_size(obj:`int`): The vocabulary size.
        num_labels(obj:`int`): All the labels that the data has.
        network_name(obj: `str`, optional, defaults to `lstm`): which network you would like.
        pad_token_id(obj:`int`, optinal, defaults to 0) : The pad token index.

    Returns:
        model(obj:`paddle.nn.Layer`): A model.
    """
    if network_name == 'bow':
        network = BoWModel(
            vocab_size, num_labels=num_labels, padding_idx=pad_token_id)
    elif network_name == 'bilstm':
        # direction choice: forward, backword, bidirectional
        network = LSTMModel(
            vocab_size=vocab_size,
            num_labels=num_labels,
            direction='bidirectional',
            padding_idx=pad_token_id)
    elif network_name == 'bigru':
        # direction choice: forward, backword, bidirectional
        network = GRUModel(
            vocab_size=vocab_size,
            num_labels=num_labels,
            direction='bidirectional',
            padding_idx=pad_token_id)
    elif network_name == 'birnn':
        # direction choice: forward, backword, bidirectional
        network = RNNModel(
            vocab_size=vocab_size,
            num_labels=num_labels,
            direction='bidirectional',
            padding_idx=pad_token_id)
    elif network_name == 'lstm':
        # direction choice: forward, backword, bidirectional
        network = LSTMModel(
            vocab_size=vocab_size,
            num_labels=num_labels,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
    elif network_name == 'gru':
        # direction choice: forward, backword, bidirectional
        network = GRUModel(
            vocab_size=vocab_size,
            num_labels=num_labels,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
    elif network_name == 'rnn':
        # direction choice: forward, backword, bidirectional
        network = RNNModel(
            vocab_size=vocab_size,
            num_labels=num_labels,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
    elif network_name == 'bilstm_attn':
        lstm_hidden_size = 196
        attention = SelfInteractiveAttention(hidden_size=2 * lstm_hidden_size)
        network = BiLSTMAttentionModel(
            attention_layer=attention,
            vocab_size=vocab_size,
            lstm_hidden_size=lstm_hidden_size,
            num_labels=num_labels,
            padding_idx=pad_token_id)
    elif network_name == 'textcnn':
        network = TextCNNModel(
            vocab_size=vocab_size,
            padding_idx=pad_token_id,
            num_labels=num_labels)
    else:
        raise ValueError(
            "Unknown network: %s, it must be one of bow, lstm, bilstm, gru, bigru, rnn, birnn, bilstm_attn and textcnn."
            % network_name)

    model = paddle.Model(network)

    return model


def convert_example(example, vocab, unk_token_id=1, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks. 
    It use `jieba.cut` to tokenize text.

    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        vocab(obj:`dict`): The vocabulary.
        unk_token_id(obj:`int`, defaults to 1): The unknown token id.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.s
        valid_length(obj:`int`): The input sequence valid length.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """

    # tokenize raw text and convert the token to ids
    # tokens_raw = ' '.join(jieba.cut(example[0]).split(' ')
    input_ids = []
    for token in jieba.cut(example[0]):
        token_id = vocab.get(token, unk_token_id)
        input_ids.append(token_id)
    valid_length = len(input_ids)

    if not is_test:
        label = np.array(example[-1], dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


if __name__ == "__main__":
    # Running config setting.
    config = RunConfig(
        save_dir=args.save_dir,
        use_gpu=args.use_gpu,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs)

    # Loads vocab.
    if not os.path.exists(args.vocab_path):
        raise RuntimeError('The vocab_path  can not be found in the path %s' %
                           args.vocab_path)
    vocab = load_vocab(args.vocab_path)
    if '[PAD]' not in vocab:
        pad_token_id = len(vocab)
        vocab['[PAD]'] = pad_token_id
    else:
        pad_token_id = vocab['[PAD]']

    # Loads dataset.
    train_dataset, dev_dataset, test_dataset = ppnlp.datasets.ChnSentiCorp.get_datasets(
        ['train', 'dev', 'test'])

    # Constructs the newtork.
    model = create_model(
        len(vocab),
        len(train_dataset.get_labels()),
        network_name=args.network_name.lower(),
        pad_token_id=pad_token_id)

    # Reads data and generates mini-batches.
    trans_fn = partial(
        convert_example,
        vocab=vocab,
        unk_token_id=vocab['[UNK]'],
        is_test=False)
    train_loader = create_dataloader(
        train_dataset,
        trans_fn=trans_fn,
        batch_size=config.batch_size,
        mode='train',
        pad_token_id=pad_token_id)
    dev_loader = create_dataloader(
        dev_dataset,
        trans_fn=trans_fn,
        batch_size=config.batch_size,
        mode='validation',
        pad_token_id=pad_token_id)
    test_loader = create_dataloader(
        test_dataset,
        trans_fn=trans_fn,
        batch_size=config.batch_size,
        mode='test',
        pad_token_id=pad_token_id)

    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=config.lr)

    # Defines loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy(name="acc_accumulation")

    model.prepare(optimizer, criterion, metric)

    # Loads pre-trained parameters.
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    # Starts training and evaluating.
    model.fit(train_loader,
              dev_loader,
              epochs=config.epochs,
              eval_freq=config.eval_freq,
              log_freq=config.eval_freq,
              save_dir=args.save_dir,
              save_freq=config.save_freq)

    # Finally tests model.
    results = model.evaluate(test_loader)
    print("Finally test acc: %.5f" % results['acc_accumulation'])
