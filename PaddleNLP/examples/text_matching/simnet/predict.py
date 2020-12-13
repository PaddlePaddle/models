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

import paddle
import paddlenlp as ppnlp

from utils import load_vocab, generate_batch, preprocess_prediction_data

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--use_gpu", type=eval, default=False, help="Whether use GPU for training, input should be True or False")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./data/term2id.dict", help="The path to vocabulary.")
parser.add_argument('--network', type=str, default="lstm", help="Which network you would like to choose bow, cnn, lstm or gru ?")
parser.add_argument("--params_path", type=str, default='./chekpoints/final.pdparams', help="The path of model parameter to be loaded.")
args = parser.parse_args()
# yapf: enable


def predict(model, data, label_map, collate_fn, batch_size=1, pad_token_id=0):
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
        pad_token_id(obj:`int`, optinal, defaults to 0) : The pad token index.

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
    model.eval()
    for batch in batches:
        queries, titles, query_seq_lens, title_seq_lens = collate_fn(
            batch, pad_token_id=pad_token_id, return_label=False)
        queries = paddle.to_tensor(queries)
        titles = paddle.to_tensor(titles)
        query_seq_lens = paddle.to_tensor(query_seq_lens)
        title_seq_lens = paddle.to_tensor(title_seq_lens)
        probs = model(queries, titles, query_seq_lens, title_seq_lens)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device("gpu") if args.use_gpu else paddle.set_device("cpu")
    # Loads vocab.
    vocab = load_vocab(args.vocab_path)
    label_map = {0: 'dissimilar', 1: 'similar'}

    # Constructs the newtork.
    model = ppnlp.models.SimNet(
        network=args.network, vocab_size=len(vocab), num_classes=len(label_map))

    # Loads model parameters.
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % args.params_path)

    # Firstly pre-processing prediction data  and then do predict.
    data = [
        ['世界上什么东西最小', '世界上什么东西最小？'],
        ['光眼睛大就好看吗', '眼睛好看吗？'],
        ['小蝌蚪找妈妈怎么样', '小蝌蚪找妈妈是谁画的'],
    ]
    examples = preprocess_prediction_data(data, vocab)
    results = predict(
        model,
        examples,
        label_map=label_map,
        batch_size=args.batch_size,
        collate_fn=generate_batch)

    for idx, text in enumerate(data):
        print('Data: {} \t Label: {}'.format(text, results[idx]))
