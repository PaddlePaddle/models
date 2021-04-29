# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import ast
import os

import paddle
import paddle.nn.functional as F
from model import SoundClassifier
from paddleaudio.datasets import ESC50
from paddleaudio.models.panns import cnn14
from paddleaudio.utils import Timer, logger

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=50, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu",
                    type=ast.literal_eval,
                    default=True,
                    help="Whether use GPU for fine-tuning. Input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers in dataloader.")
parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint', help="Directory to model checkpoint")
parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every n epoch.")
args = parser.parse_args()

if __name__ == "__main__":
    paddle.set_device('gpu') if args.use_gpu else paddle.set_device('cpu')
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()

    pretrained_model = cnn14(pretrained=True, extract_embedding=True)
    model = SoundClassifier(pretrained_model, num_class=len(ESC50.label_list))
    optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate, parameters=model.parameters())
    criterion = paddle.nn.loss.CrossEntropyLoss()

    train_ds = ESC50(mode='train', feat_type='mel_spect')
    dev_ds = ESC50(mode='dev', feat_type='mel_spect')

    train_sampler = paddle.io.DistributedBatchSampler(train_ds,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      drop_last=False)
    train_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        return_list=True,
        use_buffer_reader=True,
    )

    steps_per_epoch = len(train_sampler)
    timer = Timer(steps_per_epoch * args.epochs)
    timer.start()

    log_interval = 10
    current_epoch = 0
    for i in range(args.epochs):
        model.train()

        current_epoch += 1
        avg_loss = 0
        num_corrects = 0
        num_samples = 0
        for batch_idx, batch in enumerate(train_loader):
            feats, labels = batch
            logits = model(feats)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if isinstance(optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
                optimizer._learning_rate.step()
            model.clear_gradients()

            # calc loss
            avg_loss += loss.numpy()[0]

            # calc metrics
            preds = paddle.argmax(logits, axis=1)
            num_corrects += (preds == labels).numpy().sum()
            num_samples += feats.shape[0]

            timer.count()

            if (batch_idx + 1) % log_interval == 0 and local_rank == 0:
                lr = optimizer.get_lr()
                avg_loss /= log_interval
                avg_acc = num_corrects / num_samples

                print_msg = 'Epoch={}/{}, Step={}/{}'.format(current_epoch, args.epochs, batch_idx + 1, steps_per_epoch)
                print_msg += ' loss={:.4f}'.format(avg_loss)
                print_msg += ' acc={:.4f}'.format(avg_acc)
                print_msg += ' lr={:.6f} step/sec={:.2f} | ETA {}'.format(lr, timer.timing, timer.eta)
                logger.train(print_msg)

                avg_loss = 0
                num_corrects = 0
                num_samples = 0

        if current_epoch % args.save_interval == 0 and batch_idx + 1 == steps_per_epoch and local_rank == 0:
            dev_sampler = paddle.io.BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
            dev_loader = paddle.io.DataLoader(
                dev_ds,
                batch_sampler=dev_sampler,
                num_workers=args.num_workers,
                return_list=True,
            )

            model.eval()
            num_corrects = 0
            num_samples = 0
            with logger.processing('Evaluation on validation dataset'):
                for batch_idx, batch in enumerate(dev_loader):
                    feats, labels = batch
                    logits = model(feats)

                    preds = paddle.argmax(logits, axis=1)
                    num_corrects += (preds == labels).numpy().sum()
                    num_samples += feats.shape[0]

            print_msg = '[Evaluation result]'
            print_msg += ' dev_acc={:.4f}'.format(num_corrects / num_samples)

            logger.eval(print_msg)

            # save model
            save_dir = os.path.join(args.checkpoint_dir, 'epoch_{}'.format(current_epoch))
            logger.info('Saving model checkpoint to {}'.format(save_dir))
            paddle.save(model.state_dict(), os.path.join(save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(save_dir, 'model.pdopt'))
