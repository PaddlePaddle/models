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
from collections import namedtuple
import argparse
import os
import random
import time

import jieba
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from config import RunConfig
from data import ChnSentiCorp, convert_tokens_to_ids, load_vocab
from model import BoWModel, LSTMModel, GRUModel, RNNModel, BiLSTMAttentionModel, TextCNNModel
from model import SelfAttention, SelfInteractiveAttention

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--use_gpu", type=eval, default=False, help="Whether use GPU for training, input should be True or False")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./word_dict.txt", help="The path to vocabulary.")
parser.add_argument('--network_name', type=str, default="bilstm_attn", help="Which network you would like to choose bow, lstm, bilstm, gru, bigru, rnn, birnn, bilstm_attn, cnn and textcnn?")
parser.add_argument("--params_path", type=str, default=None, required=True, help="The path of model parameter to be loaded.")
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
    seq_lens = [entry.seq_len for entry in batch]

    batch_max_seq_len = max(seq_lens)
    texts = [entry.text for entry in batch]
    pad_texts_to_max_seq_len(texts, batch_max_seq_len, pad_token_id)

    if return_label:
        labels = [[entry.label] for entry in batch]
        return texts, seq_lens, labels
    else:
        return texts, seq_lens


def create_model(vocab_size, num_labels, network_name='bilstm', padding_idx=0):
    """
    Creats model which uses to text classification. It should be BoW, LSTM/BiLSTM, GRU/BiGRU.

    Args:
        vocab_size(obj:`int`): The vocabulary size.
        num_labels(obj:`int`): All the labels that the data has.
        network_name(obj: `str`, optional, defaults to `lstm`): which network you would like.
        padding_idx(obj:`int`, optinal, defaults to 0) : The pad token index.

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


def preprocess_prediction_data(data):
    """
    It process the prediction data as the format used as training.

    Args:
        data (obj:`List[str]`): The prediction data whose each element is  a tokenized text.

    Returns:
        examples (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).

    """
    Example = namedtuple('Example', ['text', 'seq_len'])
    examples = []
    for text in data:
        tokens = " ".join(jieba.cut(text)).split(' ')
        ids = convert_tokens_to_ids(tokens, vocab)
        example = Example(text=ids, seq_len=len(ids))
        examples.append(example)
    return examples


def predict(model, data, label_map, collate_fn, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        collate_fn(obj: `callable`): function to generate mini-batch data by merging
            the sample list.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in data:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.network.eval()
    for batch in batches:
        texts, seq_lens = collate_fn(
            batch, pad_token_id=pad_token_id, return_label=False)
        texts = paddle.to_tensor(texts)
        seq_lens = paddle.to_tensor(seq_lens)
        logits = model.network(texts, seq_lens)
        probs = F.softmax(logits, axis=-1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    # Loads vocab.
    vocab = load_vocab(args.vocab_path)
    if '[PAD]' not in vocab:
        pad_token_id = len(vocab)
        vocab['[PAD]'] = pad_token_id
    else:
        pad_token_id = vocab['[PAD]']

    label_map = {0: 'negative', 1: 'positive'}

    paddle.set_device("gpu") if args.use_gpu else paddle.set_device("cpu")

    # Constructs the newtork.
    model = create_model(
        len(vocab),
        num_labels=len(label_map),
        network_name=args.network_name.lower(),
        padding_idx=pad_token_id)

    # Loads model parameters.
    model.load(args.params_path)
    print("Loaded parameters from %s" % args.params_path)

    # Firstly pre-processing prediction data  and then do predict.
    data = [
        '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般',
        '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
        '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    ]
    examples = preprocess_prediction_data(data)
    results = predict(
        model,
        examples,
        label_map=label_map,
        batch_size=args.batch_size,
        collate_fn=generate_batch)

    for idx, text in enumerate(data):
        print('Data: {} \t Lable: {}'.format(text, results[idx]))
