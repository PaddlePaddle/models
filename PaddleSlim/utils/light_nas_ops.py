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
import paddle.fluid as fluid
import paddle
import sys
sys.path.append('..')
from models import LightNASNet
from get_ops_from_program import get_ops_from_program, write_lookup_table
import copy

batch_size = 1
image_shape = [3, 224, 224]
class_dim = 1000

NAS_FILTER_SIZE = [[18, 24, 30], [24, 32, 40], [48, 64, 80], [72, 96, 120],
                   [120, 160, 192]]
NAS_LAYERS_NUMBER = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [2, 3, 4], [2, 3, 4]]
NAS_KERNEL_SIZE = [3, 5]
NAS_FILTERS_MULTIPLIER = [3, 4, 5, 6]
NAS_SHORTCUT = [0, 1]
NAS_SE = [0, 1]


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

class LightNASModel():
    def __init__(self):
        """Get init tokens in search space.
        """
        self.init_tokens = [
            3, 1, 1, 0, 1, 0, 3, 1, 1, 0, 1, 0, 3, 1, 1, 0, 1, 0, 3, 1, 1, 0, 1,
            0, 3, 1, 1, 0, 1, 0
        ]
   
    def tokens_set(self):
        tokens = []
        # [NAS_FILTERS_MULTIPLIER, NAS_FILTER_SIZE, NAS_LAYERS_NUMBER, NAS_KERNEL_SIZE, NAS_SHORTCUT, NAS_SE] * 5
        base = [0, 0, 1, 0, 0, 0] # fixed layer number, shortcut, and se

        # fina all possible combinations
        tokens_for_bottlenecks = [[],[],[],[],[]]
        for i in range(5):
            tmp = copy.deepcopy(base)
            for j in range(len(NAS_FILTERS_MULTIPLIER)):
                tmp[0] = j
                for k in range(len(NAS_KERNEL_SIZE)):
                    tmp[3] = k
                    for m in range(len(NAS_FILTER_SIZE[i])):
                        tmp[1] = m
                        tokens_for_bottlenecks[i].append(tmp)
                        tmp = copy.deepcopy(tmp)

        for tokens0 in tokens_for_bottlenecks[0]:
            for tokens1 in tokens_for_bottlenecks[1]:
                for tokens2 in tokens_for_bottlenecks[2]:
                    for tokens3 in tokens_for_bottlenecks[3]:
                        for tokens4 in tokens_for_bottlenecks[4]:
                            tokens.append(tokens0+tokens1+tokens2+tokens3+tokens4)
        
        return tokens

    def range_table(self):
        """Get range table of current search space.
        """
        # [NAS_FILTER_SIZE, NAS_LAYERS_NUMBER, NAS_KERNEL_SIZE, NAS_FILTERS_MULTIPLIER, NAS_SHORTCUT, NAS_SE] * 5
        return [
            4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2,
            2, 4, 3, 3, 2, 2, 2
        ]

    def create_net(self, tokens=None):
        """Create a network for training by tokens.
        """
        if tokens is None:
            tokens = self.init_tokens

        bottleneck_params_list = get_bottleneck_params_list(tokens)

        startup_prog = fluid.Program()
        main_prog = fluid.Program()
        cost, acc1, acc5 = build_program(
            main_prog=main_prog,
            startup_prog=startup_prog,
            bottleneck_params_list=bottleneck_params_list)

        return startup_prog, main_prog, (cost, acc1, acc5)

def build_program(main_prog,
                  startup_prog,
                  bottleneck_params_list=None):
    with fluid.program_guard(main_prog, startup_prog):       
        with fluid.unique_name.guard():
            image = fluid.layers.data(name='image',shape=[3,224,224], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
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
            
    return avg_cost, acc_top1, acc_top5


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

if __name__=='__main__':
    model = LightNASModel()
    tokens = model.tokens_set()
    n = len(tokens)
    
    all_ops = []
    for idx in range(0, n):
        current_token = tokens[idx]
        startup_program, main_program, _ = model.create_net(current_token)
        op_params = get_ops_from_program(main_program,'results/lightnas_'+str(idx)+'.txt')
        all_ops = all_ops + op_params
        if (idx+1)%10000 == 0:
            print('current net number is: ', idx)
            print('current number of ops is:', len(all_ops))
            write_lookup_table(list(all_ops), 'results/lightnas_ops_tmp.txt')

    print('{} networks have {} ops in total'.format(n, len(all_ops)))
    write_lookup_table(all_ops, 'lightnas_ops.txt')
