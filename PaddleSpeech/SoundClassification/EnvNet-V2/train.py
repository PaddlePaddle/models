import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from model import EnvNet2
from dataset import ESC50
from lr import StepDecay
from utils import Timer

import os
import ast
import argparse
import numpy as np


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epoches for fine-tuning.")
parser.add_argument("--warmup_epoch", type=int, default=10, help="Number of epoches for warmup.")
parser.add_argument("--learning_rate", type=float, default=1e-1, help="Learning rate used to train with warmup.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--save_dir", type=str, default='./ckpt', help="Directory to save model.")
parser.add_argument("--checkpoint", type=str, default='./ckpt', help="Model checkpoint to load.")
parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every n epochs.")
parser.add_argument("--eval_interval", type=int, default=10, help="Evaluate every n epochs")
parser.add_argument("--log_interval", type=int, default=10, help="Show trainning log every n steps")
parser.add_argument("--num_workers", type=int, default=8, help="Number of threads used in dataloader.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--bc", type=ast.literal_eval, default=False, help="Whether use bc learning")

args = parser.parse_args()


if __name__ == '__main__':
    # dataset
    train_dataset = ESC50(mode='train', bc_learning=args.bc)
    eval_dataset = ESC50(mode='dev', bc_learning=args.bc)

    # model
    model = EnvNet2(ESC50.n_class)

    lr_schedule = StepDecay(learning_rate=args.learning_rate, step_size=int(0.25*args.num_epochs), warmup=args.warmup_epoch)
    optimizer = paddle.optimizer.SGD(learning_rate=lr_schedule, parameters=model.parameters())

    # loss and metric
    criterion = nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    # data loader
    train_batch_sampler = paddle.io.BatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        return_list=True,
        use_buffer_reader=True)
    eval_batch_sampler = paddle.io.BatchSampler(
        eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    eval_loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=eval_batch_sampler,
        num_workers=args.num_workers,
        return_list=True)

    # train logic
    paddle.set_device('gpu') if args.use_gpu else paddle.set_device('cpu')
    steps_per_epoch = len(train_batch_sampler)
    timer = Timer(steps_per_epoch * args.num_epochs)
    timer.start()

    current_epoch = 0
    for i in range(args.num_epochs):
        current_epoch += 1
        avg_loss = 0
        avg_acc = 0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            logits = model(inputs)
            probs = F.softmax(logits, axis=1)
            loss = criterion(logits, labels)
            correct = metric.compute(probs, labels)
            acc = metric.update(correct)

            optimizer.step()
            if not isinstance(optimizer._learning_rate, float):
                optimizer._learning_rate.step(epoch=current_epoch)
            model.clear_gradients()

            avg_loss += loss.numpy()[0]
            avg_acc += acc
            timer.count()

            if (batch_idx + 1) % args.log_interval == 0:
                lr = optimizer.get_lr()
                avg_loss /= args.log_interval
                print_msg = 'Epoch={}/{}, Step={}/{}'.format(current_epoch, args.num_epochs, batch_idx+1, steps_per_epoch)
                print_msg += ' loss={:.4f}'.format(avg_loss)

                avg_acc /= args.log_interval
                if isinstance(avg_acc, np.ndarray):
                    avg_acc = avg_acc.item()
                print_msg += ' Acc={:.4f}'.format(avg_acc)
                print_msg += ' lr={:.6f} step/sec={:.2f} | ETA {}'.format(lr, timer.timing, timer.eta)
                print(print_msg)

                avg_loss = 0
                avg_acc = 0

        # evaluation
        if current_epoch % args.eval_interval == 0 and batch_idx + 1 == steps_per_epoch:
            model.eval()
            eval_avg_loss = 0
            eval_avg_acc = 0
            steps = len(eval_batch_sampler)
            for batch_idx, batch in enumerate(eval_loader):
                inputs, labels = batch
                logits = model(inputs)
                probs = F.softmax(logits, axis=1)
                loss = criterion(logits, labels)
                correct = metric.compute(probs, labels)
                acc = metric.update(correct)

                eval_avg_loss += loss.numpy()[0]
                eval_avg_acc += acc

            eval_avg_loss /= steps
            eval_avg_acc /= steps
            if isinstance(eval_avg_acc, np.ndarray):
                eval_avg_acc = eval_avg_acc.item()

            print_msg = '[Evaluation result]'
            print_msg += ' avg_loss={:.4f}'.format(eval_avg_loss)
            print_msg += ' avg_acc={:.4f}'.format(eval_avg_acc)
            print(print_msg)

        # checkpoint saving
        if current_epoch % args.save_interval == 0:
            print('Saving model checkpoint to {}'.format(args.save_dir))
            model_params_path = os.path.join(args.save_dir, 'model.pdparams')
            optim_params_path = os.path.join(args.save_dir, 'model.pdopt')
            paddle.save(model.state_dict(), model_params_path)
            paddle.save(optimizer.state_dict(), optim_params_path)
