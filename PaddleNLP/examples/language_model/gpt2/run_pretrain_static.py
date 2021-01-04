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

import argparse
import collections
import itertools
import logging
import os
import random
import time
import h5py
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import paddle
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import GPT2Model, GPT2ForPretraining, GPT2PretrainingCriterion
from paddlenlp.transformers import GPT2Tokenizer
from paddlenlp.utils.log import logger
from data import GPT2Dataset
import lr

MODEL_CLASSES = {"gpt2-medium-en": (GPT2ForPretraining, GPT2Tokenizer), }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.", )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")

    parser.add_argument(
        "--grad_clip",
        default=0.0,
        type=float,
        help="Grad clip for the parameter.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=320000,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_rate",
        default=0.01,
        type=float,
        help="Linear warmup over warmup_steps.")

    parser.add_argument(
        "--use_amp",
        type=bool,
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--use_sharding",
        type=bool,
        default=False,
        help="Spliting the parameters to many cards.")
    parser.add_argument(
        "--use_recompute",
        type=bool,
        default=False,
        help="Using the recompute to save the memory.")
    parser.add_argument(
        "--enable_addto",
        type=bool,
        default=False,
        help="Whether to enable the addto strategy for gradient accumulation or not. This is only used for AMP training."
    )
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=1.0,
        help="The value of scale_loss for fp16.")

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    args = parser.parse_args()
    return args


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_data_holder(args):
    tokens = paddle.static.data(name="tokens", shape=[-1, -1], dtype="int64")
    loss_mask = paddle.static.data(
        name="loss_mask", shape=[-1, -1], dtype="float32")
    attention_mask = paddle.static.data(
        name="attention_mask", shape=[-1, 1, -1, -1], dtype="float32")
    position_ids = paddle.static.data(
        name="position_ids", shape=[-1, -1], dtype="int64")
    labels = paddle.static.data(name="labels", shape=[-1, -1], dtype="int64")
    return [tokens, loss_mask, attention_mask, position_ids, labels]


def create_pretrained_dataset(args, input_path, data_holders, tokenizer,
                              worker_init, places):
    train_data = GPT2Dataset(file_path=input_path, tokenizer=tokenizer)

    train_batch_sampler = paddle.io.BatchSampler(
        train_data, batch_size=args.batch_size, shuffle=False)

    train_data_loader = DataLoader(
        dataset=train_data,
        places=places,
        feed_list=data_holders,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        worker_init_fn=worker_init,
        collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()),
        return_list=False)
    return train_data_loader


def create_strategy(args):
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    build_strategy.enable_addto = args.enable_addto

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 10000
    return build_strategy, exec_strategy


def build_compiled_program(args, main_program, loss):
    build_strategy, exec_strategy = create_strategy(args)
    main_program = paddle.static.CompiledProgram(
        main_program).with_data_parallel(
            loss_name=loss.name,
            exec_strategy=exec_strategy,
            build_strategy=build_strategy)
    return main_program


def reset_program_state_dict(model, state_dict):
    scale = model.initializer_range if hasattr(model, "initializer_range")\
        else model.gpt2.config["initializer_range"]

    new_state_dict = dict()
    for n, p in state_dict.items():
        if "layer_norm" not in p.name:
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            new_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
    return new_state_dict


def copy_program_state_dict(model, static_dict, tensor_dict):
    new_state_dict = dict()
    for n, p in static_dict.items():
        new_state_dict[p.name] = tensor_dict[n]
    return new_state_dict


def dist_optimizer(args, optimizer, model):
    build_strategy, exec_strategy = create_strategy(args)

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy

    dist_strategy.fuse_grad_size_in_MB = 16
    if args.use_amp:
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            'custom_white_list': ['softmax', 'layer_norm', 'gelu'],
            'init_loss_scaling': args.scale_loss,
        }
    if args.use_sharding:
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {
            "fuse_broadcast_MB": 32,
            "hybird_dp": True,
            "sharding_group_size": 8,
        }
    if args.use_recompute:
        dist_strategy.recompute = True
        dist_strategy.recompute_configs = {
            "checkpoints": model.gpt2.checkpoints
        }

    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    return optimizer


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def do_train(args):
    # Initialize the paddle and paddle fleet execute enviroment
    paddle.enable_static()
    place = paddle.set_device("gpu")
    fleet.init(is_collective=True)

    worker_num = fleet.worker_num()
    worker_index = fleet.worker_index()

    # Create the random seed for the worker
    set_seed(args)
    worker_init = WorkerInitObj(args.seed + worker_index)

    # Define the input data in the static mode
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
    data_holders = create_data_holder(args)
    [tokens, loss_mask, attention_mask, position_ids, labels] = data_holders

    model_class, tokenizer_class = MODEL_CLASSES[args.model_name_or_path]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    config = model_class.pretrained_init_configuration[args.model_name_or_path]
    if config["vocab_size"] % 8 != 0:
        config["vocab_size"] += 8 - (config["vocab_size"] % 8)

    # create the model for the gpt model
    model = GPT2ForPretraining(GPT2Model(**config))
    criterion = GPT2PretrainingCriterion()
    preds = model(tokens, position_ids, attention_mask)
    loss = criterion(preds, labels, loss_mask)

    # Create the learning_rate sheduler and optimizer
    warmup_step = args.warmup_rate * args.max_steps
    lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
        args.learning_rate, warmup_step, args.max_steps)

    clip = None
    if args.grad_clip > 0:
        clip = paddle.nn.ClipGradByNorm(clip_norm=args.grad_clip)

    # TODO @ZHUI new fluid optimizer to use recompute
    optimizer = paddle.fluid.optimizer.Adam(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        #parameters=model.parameters(),
        #weight_decay=args.weight_decay,
        grad_clip=clip,
        # apply_decay_param_fun=lambda x: x in [
        #     p.name for n, p in model.named_parameters()
        #     if not any(nd in n for nd in ["bias", "norm"])
        # ]
    )

    if worker_num == 1 and args.use_amp:
        amp_list = paddle.fluid.contrib.mixed_precision.AutoMixedPrecisionLists(
            custom_white_list=['softmax', 'layer_norm', 'gelu'])
        optimizer = paddle.fluid.contrib.mixed_precision.decorate(
            optimizer,
            amp_list,
            init_loss_scaling=args.scale_loss,
            use_dynamic_loss_scaling=True)

    if worker_num > 1:
        # Use the fleet api to compile the distributed optimizer
        optimizer = dist_optimizer(args, optimizer, model)
    optimizer.minimize(loss)

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    state_dict = model.state_dict()
    tensor_dict = paddle.load("new_gpt2.pdparams")
    # Use the state dict to update the parameter
    reset_state_dict = copy_program_state_dict(model, state_dict, tensor_dict)
    paddle.static.set_program_state(main_program, reset_state_dict)

    if worker_num == 1:
        # Construct the compiled program
        main_program = build_compiled_program(args, main_program, loss)

    pool = ThreadPoolExecutor(1)
    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if os.path.isfile(os.path.join(args.input_dir, f))
        ]
        #files.sort()
        num_files = len(files)
        random.Random(args.seed + epoch).shuffle(files)
        f_start_id = 0

        shared_file_list = {}

        if worker_num > num_files:
            remainder = worker_num % num_files
            data_file = files[(
                f_start_id * worker_num + worker_index + remainder * f_start_id)
                              % num_files]
        else:
            data_file = files[(f_start_id * worker_num + worker_index) %
                              num_files]

        previous_file = data_file

        train_data_loader = create_pretrained_dataset(
            args, data_file, data_holders, tokenizer, worker_init,
            paddle.static.cuda_places())
        single_file = True if f_start_id + 1 == len(files) else False

        for f_id in range(f_start_id, len(files)):
            if not single_file and f_id == f_start_id:
                continue
            if worker_num > num_files:
                data_file = files[(
                    f_id * worker_num + worker_index + remainder * f_id) %
                                  num_files]
            else:
                data_file = files[(f_id * worker_num + worker_index) %
                                  num_files]

            previous_file = data_file
            dataset_future = pool.submit(create_pretrained_dataset, args,
                                         data_file, data_holders, tokenizer,
                                         worker_init,
                                         paddle.static.cuda_places())
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                loss_return = exe.run(main_program,
                                      feed=batch,
                                      fetch_list=[loss])
                # In the new 2.0 api, must call this function to change the learning_rate
                lr_scheduler.step()
                if global_step % args.logging_steps == 0:
                    if (not args.n_gpu > 1) or worker_index == 0:
                        logger.info(
                            "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                            % (global_step, epoch, step, loss_return[0],
                               args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                if global_step % args.save_steps == 0:
                    if worker_index == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # TODO(fangzeyang): Udpate the save_params to paddle.static
                        paddle.static.save_inference_model(
                            output_dir,
                            feed_vars=[
                                tokens, loss_mask, attention_mask, position_ids,
                                labels
                            ],
                            fetch_vars=[loss],
                            executor=exe)
                        tokenizer.save_pretrained(output_dir)
                if global_step >= args.max_steps:
                    del train_data_loader
                    return
            del train_data_loader
            train_data_loader, data_file = dataset_future.result(timeout=None)
        epoch += 1


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
