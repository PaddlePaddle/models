import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from model import EnvNet2
from dataset import ESC50
from lr import StepDecay
from utils import Timer, compute_eval_acc

import os
import ast
import argparse
import numpy as np

parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=1000,
    help="Number of epochs for fine-tuning.")
parser.add_argument(
    "--start_epoch", type=int, default=0, help="Starting epoch.")
parser.add_argument(
    "--warmup_epoch", type=int, default=10, help="Number of epochs for warmup.")
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-1,
    help="Learning rate used to train with warmup.")
parser.add_argument(
    "--lr_decay_step",
    type=int,
    default=300,
    help="Learning rate decay every n epochs.")
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Total examples' number in batch for training.")
parser.add_argument(
    "--save_dir", type=str, default='./ckpt', help="Directory to save model.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default='./ckpt/model.pdparams',
    help="Model checkpoint to load.")
parser.add_argument(
    "--checkpoint_op",
    type=str,
    default='./ckpt/model.pdopt',
    help="Optimizer checkpoint to load.")
parser.add_argument(
    "--save_interval",
    type=int,
    default=200,
    help="Save checkpoint every n epochs.")
parser.add_argument(
    "--eval_interval", type=int, default=10, help="Evaluate every n epochs")
parser.add_argument(
    "--log_interval",
    type=int,
    default=5,
    help="Show trainning log every n steps")
parser.add_argument(
    "--num_workers",
    type=int,
    default=32,
    help="Number of threads used in dataloader.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument(
    "--bc", type=ast.literal_eval, default=True, help="Whether use bc learning")

args = parser.parse_args()

if __name__ == '__main__':
    # train params
    if args.bc:
        args.num_epochs *= 2
        args.lr_decay_step *= 2
    print(args.__dict__)

    # device
    paddle.set_device('gpu') if args.use_gpu else paddle.set_device('cpu')

    # model
    model = EnvNet2(n_class=ESC50.n_class, checkpoint=args.checkpoint)
    paddle.summary(model, input_size=(1, 1, 1, ESC50.input_length))

    # dataset
    train_dataset = ESC50(mode='train', bc_learning=args.bc)
    eval_dataset = ESC50(mode='dev')
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        return_list=True,
        use_buffer_reader=True)
    eval_batch_sampler = paddle.io.BatchSampler(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)
    eval_loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=eval_batch_sampler,
        num_workers=1,  # only one thread to keep sample order
        return_list=True)

    # optimizer
    lr_schedule = StepDecay(
        learning_rate=args.learning_rate,
        step_size=args.lr_decay_step,
        warmup=args.warmup_epoch)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr_schedule,
        parameters=model.parameters(),
        weight_decay=5e-4,
        momentum=0.9,
        use_nesterov=True)
    if args.checkpoint_op is not None and os.path.exists(args.checkpoint_op):
        state_dict = paddle.load(args.checkpoint_op)
        optimizer.set_state_dict(state_dict)
        print('Loaded optimizer parameters from %s' %
              os.path.abspath(args.checkpoint_op))

    # loss and metric
    # criterion = nn.loss.KLDivLoss()
    criterion = nn.loss.CrossEntropyLoss(soft_label=True)
    metric = paddle.metric.Accuracy()

    # train logic
    steps_per_epoch = len(train_batch_sampler)
    timer = Timer(steps_per_epoch * args.num_epochs)
    timer.start()

    current_epoch = args.start_epoch
    for i in range(args.num_epochs):
        current_epoch += 1
        avg_loss = 0
        avg_acc = 0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            logits = model(inputs)
            probs = F.softmax(logits)
            loss = criterion(logits, labels)
            correct = metric.compute(probs, labels)
            acc = metric.update(correct)

            loss.backward()
            optimizer.step()
            if not isinstance(optimizer._learning_rate, float):
                optimizer._learning_rate.step(epoch=current_epoch)
            optimizer.clear_grad()

            avg_loss += loss.numpy()[0]
            avg_acc += acc
            timer.count()

            if (batch_idx + 1) % args.log_interval == 0:
                lr = optimizer.get_lr()
                avg_loss /= args.log_interval
                print_msg = 'Epoch={}/{}, Step={}/{}'.format(
                    current_epoch, args.num_epochs, batch_idx + 1,
                    steps_per_epoch)
                print_msg += ' loss={:.4f}'.format(avg_loss)

                avg_acc /= args.log_interval
                if isinstance(avg_acc, np.ndarray):
                    avg_acc = avg_acc.item()
                print_msg += ' acc={:.4f}'.format(avg_acc)
                print_msg += ' lr={:.6f} step/sec={:.2f} | ETA {}'.format(
                    lr, timer.timing, timer.eta)
                print(print_msg)

                avg_loss = 0
                avg_acc = 0

        # evaluation
        if current_epoch % args.eval_interval == 0 and batch_idx + 1 == steps_per_epoch:
            model.eval()
            preds = []
            targets = []
            steps = len(eval_batch_sampler)
            for batch_idx, batch in enumerate(eval_loader):
                inputs, labels = batch
                logits = model(inputs)
                probs = F.softmax(logits)
                preds.extend(probs.numpy())
                targets.extend(labels.numpy())

            eval_acc, eval_loss = compute_eval_acc(
                np.asarray(preds), np.asarray(targets), ESC50.n_crops)
            print_msg = '[Evaluation result]'
            print_msg += ' avg_loss={:.4f}'.format(eval_loss)
            print_msg += ' avg_acc={:.4f}'.format(eval_acc)
            print(print_msg)

        # checkpoint saving
        if current_epoch % args.save_interval == 0:
            print('Saving model checkpoint to {}'.format(args.save_dir))
            model_params_path = os.path.join(args.save_dir, 'model.pdparams')
            optim_params_path = os.path.join(args.save_dir, 'model.pdopt')
            paddle.save(model.state_dict(), model_params_path)
            paddle.save(optimizer.state_dict(), optim_params_path)
