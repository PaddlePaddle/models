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

import paddle
import paddlenlp as ppnlp
from paddlenlp.datasets import LCQMC

from utils import load_vocab, generate_batch, convert_example

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument('--use_gpu', type=eval, default=False, help="Whether use GPU for training, input should be True or False")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='chekpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./simnet_word_dict.txt", help="The directory to dataset.")
parser.add_argument('--network', type=str, default="lstm", help="Which network you would like to choose bow, cnn, lstm or gru ?")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
args = parser.parse_args()
# yapf: enable


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
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_sampler=sampler,
        return_list=True,
        collate_fn=lambda batch: generate_batch(batch, pad_token_id=pad_token_id))
    return dataloader


if __name__ == "__main__":
    paddle.set_device('gpu') if args.use_gpu else paddle.set_device('cpu')

    # Loads vocab.
    if not os.path.exists(args.vocab_path):
        raise RuntimeError('The vocab_path  can not be found in the path %s' %
                           args.vocab_path)
    vocab = load_vocab(args.vocab_path)

    # Loads dataset.
    train_ds, dev_dataset, test_ds = LCQMC.get_datasets(
        ['train', 'dev', 'test'])

    # Constructs the newtork.
    label_list = train_ds.get_labels()
    model = ppnlp.models.SimNet(
        network=args.network,
        vocab_size=len(vocab),
        num_classes=len(label_list))
    model = paddle.Model(model)

    # Reads data and generates mini-batches.
    trans_fn = partial(convert_example, vocab=vocab, is_test=False)
    train_loader = create_dataloader(
        train_ds, trans_fn=trans_fn, batch_size=args.batch_size, mode='train')
    dev_loader = create_dataloader(
        dev_dataset,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='validation')
    test_loader = create_dataloader(
        test_ds, trans_fn=trans_fn, batch_size=args.batch_size, mode='test')

    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=args.lr)

    # Defines loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    model.prepare(optimizer, criterion, metric)

    # Loads pre-trained parameters.
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    # Starts training and evaluating.
    model.fit(
        train_loader,
        dev_loader,
        epochs=args.epochs,
        save_dir=args.save_dir, )

    # Finally tests model.
    results = model.evaluate(test_loader)
    print("Finally test acc: %.5f" % results['acc'])
