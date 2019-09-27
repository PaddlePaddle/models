# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Light-NAS space."""
import sys
import math
from paddle.fluid.contrib.slim.nas import SearchSpace
import paddle.fluid as fluid
import paddle
sys.path.append('..')
from models import LightNASNet
import reader
from get_ops_from_program import get_ops_from_program

total_images = 1281167
lr = 0.1
num_epochs = 240
batch_size = 512
lr_strategy = "cosine_decay"
l2_decay = 4e-5
momentum_rate = 0.9
image_shape = [3, 224, 224]
class_dim = 1000

__all__ = ['LightNASSpace']

NAS_FILTER_SIZE = [[18, 24, 30], [24, 32, 40], [48, 64, 80], [72, 96, 120],
                   [120, 160, 192]]
NAS_LAYERS_NUMBER = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [2, 3, 4], [2, 3, 4]]
NAS_KERNEL_SIZE = [3, 5]
NAS_FILTERS_MULTIPLIER = [3, 4, 5, 6]
NAS_SHORTCUT = [0, 1]
NAS_SE = [0, 1]
LATENCY_LOOKUP_TABLE_PATH = None


def get_bottleneck_params_list(var):
    """Get bottleneck_params_list from var.
    Args:
        var: list, variable list.
    Returns:
        list, bottleneck_params_list.
    """
    params_list = [
        1, 16, 1, 1, 3, 1, 0, \
        6, 24, 2, 2, 3, 1, 0, \
        6, 32, 3, 2, 3, 1, 0, \
        6, 64, 4, 2, 3, 1, 0, \
        6, 96, 3, 1, 3, 1, 0, \
        6, 160, 3, 2, 3, 1, 0, \
        6, 320, 1, 1, 3, 1, 0, \
    ]
    for i in range(5):
        params_list[i * 7 + 7] = NAS_FILTERS_MULTIPLIER[var[i * 6]]
        params_list[i * 7 + 8] = NAS_FILTER_SIZE[i][var[i * 6 + 1]]
        params_list[i * 7 + 9] = NAS_LAYERS_NUMBER[i][var[i * 6 + 2]]
        params_list[i * 7 + 11] = NAS_KERNEL_SIZE[var[i * 6 + 3]]
        params_list[i * 7 + 12] = NAS_SHORTCUT[var[i * 6 + 4]]
        params_list[i * 7 + 13] = NAS_SE[var[i * 6 + 5]]
    return params_list


def ops_of_inverted_residual_unit(in_c,
                                  in_shape,
                                  expansion,
                                  kernels,
                                  num_filters,
                                  s,
                                  ifshortcut=True,
                                  ifse=True):
    """Get ops of possible repeated inverted residual unit
    Args:
        in_c: list, a list of numbers of input channels
        in_shape: int, size of input feature map
        expansion: int, expansion factor
        kernels: list, a list of possible kernel size
        s: int, stride of depthwise conv
        ifshortcut: bool
        ifse: bool
    Returns:
        op_params: list, a list of op params
    """
    op_params = []
    for c in in_c:
        for t in expansion:
            # expansion
            op_params.append(('conv', 0, 0, 1, c, in_shape, in_shape, c * t, 1,
                              1, 0, 1, 1))
            op_params.append(('batch_norm', 'None', 1, c * t, in_shape,
                              in_shape))
            op_params.append(('activation', 'relu6', 1, c * t, in_shape,
                              in_shape))

            # depthwise
            for k in kernels:
                op_params.append(('conv', 0, 0, 1, c * t, in_shape, in_shape,
                                  c * t, c * t, k, int(int(k - 1) / 2), s, 1))
            op_params.append(('batch_norm', 'None', 1, c * t, int(in_shape / s),
                              int(in_shape / s)))
            op_params.append(('activation', 'relu6', 1, c * t,
                              int(in_shape / s), int(in_shape / s)))

            # shrink
            for out_c in num_filters:
                op_params.append(('conv', 0, 0, 1, c * t, int(in_shape / s),
                                  int(in_shape / s), out_c, 1, 1, 0, 1, 1))
                op_params.append(('batch_norm', 'None', 1, out_c,
                                  int(in_shape / s), int(in_shape / s)))

                # shortcut
                if ifshortcut:
                    op_params.append(('eltwise', 2, 1, out_c, int(in_shape / s),
                                      int(in_shape / s)))
                if ifse:
                    op_params.append(('pooling', 1, 1, out_c, int(in_shape / s),
                                      int(in_shape / s), 0, 0, 1, 0, 3))
                    op_params.append(('conv', 0, 0, 1, out_c, 1, 1,
                                      int(out_c / 4), 1, 1, 0, 1, 1))
                    op_params.append(('eltwise', 2, 1, int(out_c / 4), 1, 1))
                    op_params.append(
                        ('activation', 'relu', 1, int(out_c / 4), 1, 1))
                    op_params.append(('conv', 0, 0, 1, int(out_c / 4), 1, 1,
                                      out_c, 1, 1, 0, 1, 1))
                    op_params.append(('eltwise', 2, 1, out_c, 1, 1))
                    op_params.append(('activation', 'sigmoid', 1, out_c, 1, 1))
                    op_params.append(('eltwise', 1, 1, out_c, int(in_shape / s),
                                      int(in_shape / s)))
                    op_params.append(('activation', 'relu', 1, out_c,
                                      int(in_shape / s), int(in_shape / s)))

    return op_params


def get_all_ops(ifshortcut=True, ifse=True, strides=[1, 2, 2, 2, 1, 2, 1]):
    """Get all possible ops of current search space
    Args:
        ifshortcut: bool, shortcut or not
        ifse: bool, se or not
        strides: list, list of strides for bottlenecks
    Returns:
        op_params: list, a list of all possible params
    """
    op_params = []
    # conv1_1
    op_params.append(('conv', 0, 0, 1, image_shape[0], image_shape[1],
                      image_shape[2], 32, 1, 3, 1, 2, 1))
    op_params.append(('batch_norm', 'None', 1, 32, int(image_shape[1] / 2),
                      int(image_shape[2] / 2)))
    op_params.append(('activation', 'relu6', 1, 32, int(image_shape[1] / 2),
                      int(image_shape[2] / 2)))

    # bottlenecks, TODO: different h and w for images
    in_c, in_shape = [32], int(image_shape[1] / 2)
    for i in range(len(NAS_FILTER_SIZE) + 2):
        if i == 0:
            expansion, kernels, num_filters, s = [1], [3], [16], strides[i]
        elif i == len(NAS_FILTER_SIZE) + 1:
            expansion, kernels, num_filters, s = [6], [3], [320], strides[i]
        else:
            expansion, kernels, num_filters, s = NAS_FILTERS_MULTIPLIER, \
                                                 NAS_KERNEL_SIZE, \
                                                 NAS_FILTER_SIZE[i-1], \
                                                 strides[i]

        # first block
        tmp_ops = ops_of_inverted_residual_unit(
            in_c, in_shape, expansion, kernels, num_filters, s, False, ifse)
        op_params = op_params + tmp_ops

        in_c, in_shape = num_filters, int(in_shape / s)

        # repeated block: possibly more ops, but it is ok
        tmp_ops = ops_of_inverted_residual_unit(in_c, in_shape, expansion,
                                                kernels, num_filters, 1,
                                                ifshortcut, ifse)
        op_params = op_params + tmp_ops

    # last conv
    op_params.append(('conv', 0, 0, 1, 320, in_shape, in_shape, 1280, 1, 1, 0,
                      1, 1))
    op_params.append(('batch_norm', 'None', 1, 1280, in_shape, in_shape))
    op_params.append(('activation', 'relu6', 1, 1280, in_shape, in_shape))
    op_params.append(('pooling', 1, 1, 1280, in_shape, in_shape, in_shape, 0, 1,
                      0, 3))
    # fc, converted to 1x1 conv
    op_params.append(('conv', 0, 0, 1, 1280, 1, 1, class_dim, 1, 1, 0, 1, 1))
    op_params.append(('eltwise', 2, 1, 1000, 1, 1))

    op_params.append(('softmax', -1, 1, 1000, 1, 1))
    op_params.append(('eltwise', 1, 1, 1, 1, 1))
    op_params.append(('eltwise', 2, 1, 1, 1, 1))
    return list(set(op_params))


class LightNASSpace(SearchSpace):
    def __init__(self):
        super(LightNASSpace, self).__init__()
        if LATENCY_LOOKUP_TABLE_PATH:
            self.init_latency_lookup_table(LATENCY_LOOKUP_TABLE_PATH)

    def init_latency_lookup_table(self, latency_lookup_table_path):
        """Init lookup table.
        Args:
            latency_lookup_table_path: str, latency lookup table path.
        """
        self._latency_lookup_table = dict()
        for line in open(latency_lookup_table_path):
            line = line.split()
            self._latency_lookup_table[tuple(line[:-1])] = float(line[-1])

    def init_tokens(self):
        """Get init tokens in search space.
        """
        return [
            3, 1, 1, 0, 1, 0, 3, 1, 1, 0, 1, 0, 3, 1, 1, 0, 1, 0, 3, 1, 1, 0, 1,
            0, 3, 1, 1, 0, 1, 0
        ]

    def range_table(self):
        """Get range table of current search space.
        """
        # [NAS_FILTER_SIZE, NAS_LAYERS_NUMBER, NAS_KERNEL_SIZE, NAS_FILTERS_MULTIPLIER, NAS_SHORTCUT, NAS_SE] * 5
        return [
            4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2,
            2, 4, 3, 3, 2, 2, 2
        ]

    def get_model_latency(self, program):
        """Get model latency according to program.
        Args:
            program(Program): The program to get latency.
        Return:
            (float): model latency.
        """
        ops = get_ops_from_program(program)
        latency = sum(
            [self._latency_lookup_table[tuple(map(str, op))] for op in ops])
        return latency

    def create_net(self, tokens=None):
        """Create a network for training by tokens.
        """
        if tokens is None:
            tokens = self.init_tokens()

        bottleneck_params_list = get_bottleneck_params_list(tokens)

        startup_prog = fluid.Program()
        train_prog = fluid.Program()
        test_prog = fluid.Program()
        train_py_reader, train_cost, train_acc1, train_acc5, global_lr = build_program(
            is_train=True,
            main_prog=train_prog,
            startup_prog=startup_prog,
            bottleneck_params_list=bottleneck_params_list)
        test_py_reader, test_cost, test_acc1, test_acc5 = build_program(
            is_train=False,
            main_prog=test_prog,
            startup_prog=startup_prog,
            bottleneck_params_list=bottleneck_params_list)
        test_prog = test_prog.clone(for_test=True)
        train_batch_size = batch_size / 4
        test_batch_size = batch_size
        train_reader = paddle.batch(
            reader.train(), batch_size=train_batch_size, drop_last=True)
        test_reader = paddle.batch(reader.val(), batch_size=test_batch_size)

        with fluid.program_guard(train_prog, startup_prog):
            train_py_reader.decorate_paddle_reader(train_reader)

        with fluid.program_guard(test_prog, startup_prog):
            test_py_reader.decorate_paddle_reader(test_reader)
        return startup_prog, train_prog, test_prog, (
            train_cost, train_acc1, train_acc5,
            global_lr), (test_cost, test_acc1,
                         test_acc5), train_py_reader, test_py_reader


def build_program(is_train,
                  main_prog,
                  startup_prog,
                  bottleneck_params_list=None):
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(py_reader)
            model = LightNASNet()
            avg_cost, acc_top1, acc_top5 = net_config(
                image,
                label,
                model,
                class_dim=class_dim,
                bottleneck_params_list=bottleneck_params_list,
                scale_loss=1.0)

            avg_cost.persistable = True
            acc_top1.persistable = True
            acc_top5.persistable = True
            if is_train:
                params = model.params
                params["total_images"] = total_images
                params["lr"] = lr
                params["num_epochs"] = num_epochs
                params["learning_strategy"]["batch_size"] = batch_size
                params["learning_strategy"]["name"] = lr_strategy
                params["l2_decay"] = l2_decay
                params["momentum_rate"] = momentum_rate
                optimizer = optimizer_setting(params)
                optimizer.minimize(avg_cost)
                global_lr = optimizer._global_learning_rate()

        if is_train:
            return py_reader, avg_cost, acc_top1, acc_top5, global_lr
        else:
            return py_reader, avg_cost, acc_top1, acc_top5


def net_config(image,
               label,
               model,
               class_dim=1000,
               bottleneck_params_list=None,
               scale_loss=1.0):
    bottleneck_params_list = [
        bottleneck_params_list[i:i + 7]
        for i in range(0, len(bottleneck_params_list), 7)
    ]
    out = model.net(input=image,
                    bottleneck_params_list=bottleneck_params_list,
                    class_dim=class_dim)
    cost, pred = fluid.layers.softmax_with_cross_entropy(
        out, label, return_softmax=True)
    if scale_loss > 1:
        avg_cost = fluid.layers.mean(x=cost) * float(scale_loss)
    else:
        avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)
    return avg_cost, acc_top1, acc_top5


def optimizer_setting(params):
    """optimizer setting.
    Args:
        params: dict, params.
    """
    ls = params["learning_strategy"]
    l2_decay = params["l2_decay"]
    momentum_rate = params["momentum_rate"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)
        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "cosine_warmup_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        step = int(math.ceil(float(total_images) / batch_size))
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay_with_warmup(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "exponential_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        num_epochs = params["num_epochs"]
        start_lr = params["lr"]
        total_step = int((total_images / batch_size) * num_epochs)
        decay_step = int((total_images / batch_size) * 2.4)
        lr = start_lr
        lr = fluid.layers.exponential_decay(
            learning_rate=start_lr,
            decay_steps=decay_step,
            decay_rate=0.97,
            staircase=True)
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=lr)
    elif ls["name"] == "exponential_decay_with_RMSProp":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(math.ceil(float(total_images) / batch_size))
        decay_step = int(2.4 * step)
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        optimizer = fluid.optimizer.RMSProp(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=lr,
                decay_steps=decay_step,
                decay_rate=0.97,
                staircase=False),
            regularization=fluid.regularizer.L2Decay(l2_decay),
            momentum=0.9,
            rho=0.9,
            epsilon=0.001)
    elif ls["name"] == "linear_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        num_epochs = params["num_epochs"]
        start_lr = params["lr"]
        end_lr = 0
        total_step = int((total_images / batch_size) * num_epochs)
        lr = fluid.layers.polynomial_decay(
            start_lr, total_step, end_lr, power=1)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "adam":
        lr = params["lr"]
        optimizer = fluid.optimizer.Adam(learning_rate=lr)
    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    return optimizer
