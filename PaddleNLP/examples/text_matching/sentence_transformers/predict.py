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

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad

from model import SentenceTransformer

MODEL_CLASSES = {
    "bert": (ppnlp.transformers.BertModel, ppnlp.transformers.BertTokenizer),
    'ernie': (ppnlp.transformers.ErnieModel, ppnlp.transformers.ErnieTokenizer),
    'roberta': (ppnlp.transformers.RobertaModel,
                ppnlp.transformers.RobertaTokenizer),
    'electra': (ppnlp.transformers.Electra, ppnlp.transformers.ElectraTokenizer)
}


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_type", default='ernie', type=str, help="Model type selected in the list: " +", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name", default='ernie-1.0', type=str, help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(sum([list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values()], [])))
    parser.add_argument("--params_path", type=str, default='./checkpoint/model_4900/model_state.pdparams', help="The path to model parameters to be loaded.")

    parser.add_argument("--max_seq_length", default=50, type=int, help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--n_gpu", type=int, default=0, help="Number of GPUs to use, 0 for CPU.")
    args = parser.parse_args()
    return args
# yapf: enable


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``
    - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A BERT sequence pair mask has the following format:
    ::
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If only one sequence, only returns the first portion of the mask (0's).


    Args:
        example(obj:`list[str]`): List of input data, containing query, title and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_list(obj:`list[str]`): All the labels that the data has.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        query_input_ids(obj:`list[int]`): The list of query token ids.
        query_segment_ids(obj: `list[int]`): List of query sequence pair mask.
        title_input_ids(obj:`list[int]`): The list of title token ids.
        title_segment_ids(obj: `list[int]`): List of title sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    print(example)
    query, title = example[0], example[1]

    query_encoded_inputs = tokenizer.encode(
        text=query, max_seq_len=max_seq_length)
    query_input_ids = query_encoded_inputs["input_ids"]
    query_segment_ids = query_encoded_inputs["segment_ids"]

    title_encoded_inputs = tokenizer.encode(
        text=title, max_seq_len=max_seq_length)
    title_input_ids = title_encoded_inputs["input_ids"]
    title_segment_ids = title_encoded_inputs["segment_ids"]

    if not is_test:
        # create label maps if classification task
        label = example[-1]
        label_map = {}
        for (i, l) in enumerate(label_list):
            label_map[l] = i
        label = label_map[label]
        label = np.array([label], dtype="int64")
        return query_input_ids, query_segment_ids, title_input_ids, title_segment_ids, label
    else:
        return query_input_ids, query_segment_ids, title_input_ids, title_segment_ids


def predict(model, data, tokenizer, label_map, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text_pair in data:
        query_input_ids, query_segment_ids, title_input_ids, title_segment_ids = convert_example(
            text_pair,
            tokenizer,
            label_list=label_map.values(),
            max_seq_length=args.max_seq_length,
            is_test=True)
        examples.append((query_input_ids, query_segment_ids, title_input_ids,
                         title_segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tilte_segment
    ): [data for data in fn(samples)]

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        query_input_ids, query_segment_ids, title_input_ids, title_segment_ids = batchify_fn(
            batch)

        query_input_ids = paddle.to_tensor(query_input_ids)
        query_segment_ids = paddle.to_tensor(query_segment_ids)
        title_input_ids = paddle.to_tensor(title_input_ids)
        title_segment_ids = paddle.to_tensor(title_segment_ids)

        probs = model(
            query_input_ids,
            title_input_ids,
            query_token_type_ids=query_segment_ids,
            title_token_type_ids=title_segment_ids)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    args = parse_args()
    paddle.set_device("gpu" if args.n_gpu else "cpu")

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.model_name == 'ernie_tiny':
        # ErnieTinyTokenizer is special for ernie_tiny pretained model.
        tokenizer = ppnlp.transformers.ErnieTinyTokenizer.from_pretrained(
            args.model_name)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_name)

    data = [
        ['世界上什么东西最小', '世界上什么东西最小？'],
        ['光眼睛大就好看吗', '眼睛好看吗？'],
        ['小蝌蚪找妈妈怎么样', '小蝌蚪找妈妈是谁画的'],
    ]
    label_map = {0: 'dissimilar', 1: 'similar'}

    pretrained_model = model_class.from_pretrained(args.model_name)
    model = SentenceTransformer(pretrained_model)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = predict(
        model, data, tokenizer, label_map, batch_size=args.batch_size)
    for idx, text in enumerate(data):
        print('Data: {} \t Lable: {}'.format(text, results[idx]))
