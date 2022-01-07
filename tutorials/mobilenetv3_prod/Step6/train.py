import datetime
import os
import sys
import time

import paddle
from paddle import nn
from paddle.static import InputSpec
import paddlevision

import presets
import utils

import numpy as np
import random

apex = None

import numpy as np
from reprod_log import ReprodLogger


def train_one_epoch(model,
                    criterion,
                    optimizer,
                    data_loader,
                    epoch,
                    print_freq,
                    amp_level=None,
                    scaler=None):
    model.train()
    # training log
    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    acc1 = 0.0
    acc5 = 0.0
    reader_start = time.time()
    batch_past = 0

    for batch_idx, (image, target) in enumerate(data_loader):
        train_reader_cost += time.time() - reader_start
        train_start = time.time()

        if amp_level is not None:
            with paddle.amp.auto_cast(level=amp_level):
                output = model(image)
                loss = criterion(output, target)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
        else:
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        optimizer.clear_grad()
        train_run_cost += time.time() - train_start
        acc = utils.accuracy(output, target, topk=(1, 5))
        acc1 += acc[0].item()
        acc5 += acc[1].item()
        total_samples += image.shape[0]
        batch_past += 1

        if batch_idx % print_freq == 0:
            msg = "[Epoch {}, iter: {}] top1: {:.5f}, top5: {:.5f}, lr: {:.5f}, loss: {:.5f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {}, avg_ips: {:.5f} images/sec.".format(
                epoch, batch_idx, acc1 / batch_past, acc5 / batch_past,
                optimizer.get_lr(),
                loss.item(), train_reader_cost / batch_past,
                (train_reader_cost + train_run_cost) / batch_past,
                total_samples / batch_past,
                total_samples / (train_reader_cost + train_run_cost))
            if paddle.distributed.get_rank() <= 0:
                print(msg)
                sys.stdout.flush()
            train_reader_cost = 0.0
            train_run_cost = 0.0
            total_samples = 0
            acc1 = 0.0
            acc5 = 0.0
            batch_past = 0

        reader_start = time.time()


def evaluate(model, criterion, data_loader, print_freq=100, amp_level=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with paddle.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq,
                                                     header):
            if amp_level is not None:
                with paddle.amp.auto_cast(level=amp_level):
                    output = model(image)
                    loss = criterion(output, target)
            else:
                output = model(image)
                loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'.format(
        top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (
        256, 224)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = paddlevision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=crop_size,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob))

    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = paddlevision.datasets.ImageFolder(
        valdir,
        presets.ClassificationPresetEval(
            crop_size=crop_size, resize_size=resize_size))

    print("Creating data loaders")
    train_sampler = paddle.io.DistributedBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)
    # train_sampler = paddle.io.RandomSampler(dataset)

    test_sampler = paddle.io.SequenceSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    try:
        paddle.set_device(args.device)
    except:
        print("device set error, use default device...")

    # multi cards
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args)
    train_batch_sampler = train_sampler
    # train_batch_sampler = paddle.io.BatchSampler(
    #     sampler=train_sampler, batch_size=args.batch_size)
    data_loader = paddle.io.DataLoader(
        dataset=dataset,
        num_workers=args.workers,
        return_list=True,
        batch_sampler=train_batch_sampler)
    test_batch_sampler = paddle.io.BatchSampler(
        sampler=test_sampler, batch_size=args.batch_size)
    data_loader_test = paddle.io.DataLoader(
        dataset_test,
        batch_sampler=test_batch_sampler,
        num_workers=args.workers)

    print("Creating model")
    model = paddlevision.models.__dict__[args.model](
        pretrained=args.pretrained)

    if args.pact_quant:
        try:
            from paddleslim.dygraph.quant import QAT
        except Exception as e:
            print(
                'Unable to QAT, please install paddleslim, for example: `pip install paddleslim`'
            )
            return

        quant_config = {
            # activation preprocess type, default is None and no preprocessing is performed.
            'activation_preprocess_type': 'PACT',
            # weight preprocess type, default is None and no preprocessing is performed. 
            'weight_preprocess_type': None,
            # weight quantize type, default is 'channel_wise_abs_max'
            'weight_quantize_type': 'channel_wise_abs_max',
            # activation quantize type, default is 'moving_average_abs_max'
            'activation_quantize_type': 'moving_average_abs_max',
            # weight quantize bit num, default is 8
            'weight_bits': 8,
            # activation quantize bit num, default is 8
            'activation_bits': 8,
            # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
            'dtype': 'int8',
            # window size for 'range_abs_max' quantization. default is 10000
            'window_size': 10000,
            # The decay coefficient of moving average, default is 0.9
            'moving_rate': 0.9,
            # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
            'quantizable_layer_type': ['Conv2D', 'Linear'],
        }

        quanter = QAT(config=quant_config)
        quanter.quantize(model)
        print("Quanted model")

    criterion = nn.CrossEntropyLoss()

    lr_scheduler = paddle.optimizer.lr.StepDecay(
        args.lr, step_size=args.lr_step_size, gamma=args.lr_gamma)

    opt_name = args.opt.lower()
    if opt_name == 'sgd':
        optimizer = paddle.optimizer.Momentum(
            learning_rate=lr_scheduler,
            momentum=args.momentum,
            parameters=model.parameters(),
            weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = paddle.optimizer.RMSprop(
            learning_rate=lr_scheduler,
            momentum=args.momentum,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9)
    else:
        raise RuntimeError(
            "Invalid optimizer {}. Only SGD and RMSprop are supported.".format(
                args.opt))

    if args.resume:
        layer_state_dict = paddle.load(os.path.join(args.resume, '.pdparams'))
        model.set_state_dict(layer_state_dict)
        opt_state_dict = paddle.load(os.path.join(args.resume, '.pdopt'))
        optimizer.load_state_dict(opt_state_dict)

    scaler = None
    if args.amp_level is not None:
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        if args.amp_level == 'O2':
            model = paddle.amp.decorate(models=model, level='O2')

    # multi cards
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.test_only and paddle.distributed.get_rank() == 0:
        top1 = evaluate(
            model, criterion, data_loader_test, amp_level=args.amp_level)
        return top1

    print("Start training")
    start_time = time.time()
    best_top1 = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, epoch,
                        args.print_freq, args.amp_level, scaler)
        lr_scheduler.step()
        if paddle.distributed.get_rank() == 0:
            top1 = evaluate(
                model, criterion, data_loader_test, amp_level=args.amp_level)
            if args.output_dir:
                paddle.save(model.state_dict(),
                            os.path.join(args.output_dir,
                                         'model_{}.pdparams'.format(epoch)))
                paddle.save(optimizer.state_dict(),
                            os.path.join(args.output_dir,
                                         'model_{}.pdopt'.format(epoch)))
                paddle.save(model.state_dict(),
                            os.path.join(args.output_dir, 'latest.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(args.output_dir, 'latest.pdopt'))
                if top1 > best_top1:
                    best_top1 = top1
                    paddle.save(model.state_dict(),
                                os.path.join(args.output_dir, 'best.pdparams'))
                    paddle.save(optimizer.state_dict(),
                                os.path.join(args.output_dir, 'best.pdopt'))

                    if args.pact_quant:
                        input_spec = [
                            InputSpec(
                                shape=[None, 3, 224, 224], dtype='float32')
                        ]
                        quanter.save_quantized_model(
                            model,
                            os.path.join(args.output_dir, "qat_inference"),
                            input_spec=input_spec)
                        print("QAT inference model saved in {args.output_dir}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return best_top1


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument('--data-path', default='./data', help='dataset')
    parser.add_argument('--model', default='mobilenet_v3_small', help='model')
    parser.add_argument('--device', default='gpu', help='device')
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument(
        '--epochs',
        default=90,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '--amp_level', default=None, help='amp level can set to be : O1/O2')
    parser.add_argument(
        '-j',
        '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument(
        '--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument(
        '--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument(
        '--wd',
        '--weight-decay',
        default=1e-4,
        type=float,
        metavar='W',
        help='weight decay (default: 1e-4)',
        dest='weight_decay')
    parser.add_argument(
        '--lr-step-size',
        default=30,
        type=int,
        help='decrease lr every step-size epochs')
    parser.add_argument(
        '--lr-gamma',
        default=0.1,
        type=float,
        help='decrease lr by a factor of lr-gamma')
    parser.add_argument(
        '--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        '--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true", )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true", )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo")
    parser.add_argument(
        '--auto-augment',
        default=None,
        help='auto augment policy (default: None)')
    parser.add_argument(
        '--random-erase',
        default=0.0,
        type=float,
        help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument(
        '--apex',
        action='store_true',
        help='Use apex for mixed precision training')
    parser.add_argument(
        '--apex-opt-level',
        default='O1',
        type=str,
        help='For apex mixed precision training'
        'O0 for FP32 training, O1 for mixed precision training.'
        'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
    )

    # Quant aware training parameters
    parser.add_argument(
        '--pact_quant',
        action='store_true',
        help='Use pact for quant aware training')

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    top1 = main(args)
    if paddle.distributed.get_rank() == 0:
        reprod_logger = ReprodLogger()
        reprod_logger.add("top1", np.array([top1]))
        reprod_logger.save("train_align_paddle.npy")
