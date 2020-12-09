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
import jieba
import numpy as np


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


def convert_ids_to_tokens(wids, inversed_vocab):
    """ Converts a token string (or a sequence of tokens) in a single integer id
        (or a sequence of ids), using the vocabulary.
    """
    tokens = []
    for wid in wids:
        wstr = inversed_vocab.get(wid, None)
        if wstr:
            tokens.append(wstr)
    return tokens


def convert_tokens_to_ids(tokens, vocab):
    """ Converts a token id (or a sequence of id) in a token string
        (or a sequence of tokens), using the vocabulary.
    """

    ids = []
    unk_id = vocab.get('[UNK]', None)
    for token in tokens:
        wid = vocab.get(token, unk_id)
        if wid:
            ids.append(wid)
    return ids


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
    queries = [entry[0] for entry in batch]
    titles = [entry[1] for entry in batch]
    query_seq_lens = [entry[2] for entry in batch]
    title_seq_lens = [entry[3] for entry in batch]

    query_batch_max_seq_len = max(query_seq_lens)
    pad_texts_to_max_seq_len(queries, query_batch_max_seq_len, pad_token_id)
    title_batch_max_seq_len = max(title_seq_lens)
    pad_texts_to_max_seq_len(titles, title_batch_max_seq_len, pad_token_id)

    if return_label:
        labels = [entry[-1] for entry in batch]
        return queries, titles, query_seq_lens, title_seq_lens, labels
    else:
        return queries, titles, query_seq_lens, title_seq_lens


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
        query_ids(obj:`list[int]`): The list of query ids.
        title_ids(obj:`list[int]`): The list of title ids.
        query_seq_len(obj:`int`): The input sequence query length.
        title_seq_len(obj:`int`): The input sequence title length.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """

    query, title = example[0], example[1]
    query_tokens = jieba.lcut(query)
    title_tokens = jieba.lcut(title)

    query_ids = convert_tokens_to_ids(query_tokens, vocab)
    query_seq_len = len(query_ids)
    title_ids = convert_tokens_to_ids(title_tokens, vocab)
    title_seq_len = len(title_ids)

    if not is_test:
        label = np.array(example[-1], dtype="int64")
        return query_ids, title_ids, query_seq_len, title_seq_len, label
    else:
        return query_ids, title_ids, query_seq_len, title_seq_len


def preprocess_prediction_data(data, vocab):
    """
    It process the prediction data as the format used as training.

    Args:
        data (obj:`List[List[str, str]]`): 
            The prediction data whose each element is a text pair. 
            Each text will be tokenized by jieba.lcut() function.

    Returns:
        examples (obj:`list`): The processed data whose each element 
            is a `list` object, which contains 

            - query_ids(obj:`list[int]`): The list of query ids.
            - title_ids(obj:`list[int]`): The list of title ids.
            - query_seq_len(obj:`int`): The input sequence query length.
            - title_seq_len(obj:`int`): The input sequence title length.

    """
    examples = []
    for query, title in data:
        query_tokens = jieba.lcut(query)
        title_tokens = jieba.lcut(title)
        query_ids = convert_tokens_to_ids(query_tokens, vocab)
        title_ids = convert_tokens_to_ids(title_tokens, vocab)
        examples.append([query_ids, title_ids, len(query_ids), len(title_ids)])
    return examples
