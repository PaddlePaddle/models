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
import math


class Lr(object):
    """
    示例：使用poly策略， 有热身，
     lr_scheduler = Lr(lr_policy='poly', base_lr=0.003, epoch_nums=200, step_per_epoch=20,
                      warm_up=True, warmup_epoch=11)
     lr = lr_scheduler.get_lr()

    示例：使用cosine策略， 有热身，
    lr_scheduler = Lr(lr_policy='cosine', base_lr=0.003, epoch_nums=200, step_per_epoch=20,
                      warm_up=True, warmup_epoch=11)
    lr = lr_scheduler.get_lr()

    示例：使用piecewise策略， 有热身，必须设置边界（decay_epoch list), gamma系数默认0.1
    lr_scheduler = Lr(lr_policy='piecewise', base_lr=0.003, epoch_nums=200, step_per_epoch=20,
                      warm_up=True, warmup_epoch=11, decay_epoch=[50], gamma=0.1)
    lr = lr_scheduler.get_lr()
    """
    def __init__(self, lr_policy, base_lr, epoch_nums, step_per_epoch,
                 power=0.9, end_lr=0.0, gamma=0.1, decay_epoch=[],
                 warm_up=False, warmup_epoch=0):
        support_lr_policy = ['poly', 'piecewise', 'cosine']
        assert lr_policy in support_lr_policy, "Only support poly, piecewise, cosine"
        self.lr_policy = lr_policy  # 学习率衰减策略 : str(`cosine`, `poly`, `piecewise`)

        assert base_lr >= 0, "Start learning rate should greater than 0"
        self.base_lr = base_lr  # 基础学习率: float

        assert end_lr >= 0, "End learning rate should greater than 0"
        self.end_lr = end_lr  # 学习率终点: float

        assert epoch_nums, "epoch_nums should greater than 0"
        assert step_per_epoch, "step_per_epoch should greater than 0"

        self.epoch_nums = epoch_nums  # epoch数: int
        self.step_per_epoch = step_per_epoch  # 每个epoch的迭代数: int
        self.total_step = epoch_nums * step_per_epoch  # 总的迭代数 :auto
        self.power = power  # 指数: float
        self.gamma = gamma  # 分段衰减的系数: float
        self.decay_epoch = decay_epoch  # 分段衰减的epoch: list
        if self.lr_policy == 'piecewise':
            assert len(decay_epoch) >= 1, "use piecewise policy, should set decay_epoch list"
        self.warm_up = warm_up  # 是否热身：bool
        if self.warm_up:
            assert warmup_epoch, "warmup_epoch should greater than 0"
            assert warmup_epoch < epoch_nums, "warmup_epoch should less than epoch_nums"
        self.warmup_epoch = warmup_epoch
        self.warmup_steps = warmup_epoch * step_per_epoch  # 热身steps：int(epoch*step_per_epoch)

    def _piecewise_decay(self):
        gamma = self.gamma
        bd = [self.step_per_epoch * e for e in self.decay_epoch]
        lr = [self.base_lr * (gamma ** i) for i in range(len(bd) + 1)]
        decayed_lr = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
        return decayed_lr

    def _poly_decay(self):
        decayed_lr = fluid.layers.polynomial_decay(
            self.base_lr, self.total_step, end_learning_rate=self.end_lr, power=self.power)
        return decayed_lr

    def _cosine_decay(self):
        decayed_lr = fluid.layers.cosine_decay(
            self.base_lr, self.step_per_epoch, self.epoch_nums)
        return decayed_lr

    def get_lr(self):
        if self.lr_policy.lower() == 'poly':
            if self.warm_up:
                warm_up_end_lr = (self.base_lr - self.end_lr) * pow(
                    (1 - self.warmup_steps / self.total_step), self.power) + self.end_lr
                print('poly warm_up_end_lr：', warm_up_end_lr)
                decayed_lr = fluid.layers.linear_lr_warmup(self._poly_decay(),
                                                           warmup_steps=self.warmup_steps,
                                                           start_lr=0.0,
                                                           end_lr=warm_up_end_lr)
            else:
                decayed_lr = self._poly_decay()
        elif self.lr_policy.lower() == 'piecewise':
            if self.warm_up:
                assert self.warmup_steps < self.decay_epoch[0] * self.step_per_epoch
                warm_up_end_lr = self.base_lr
                print('piecewise warm_up_end_lr：', warm_up_end_lr)
                decayed_lr = fluid.layers.linear_lr_warmup(self._piecewise_decay(),
                                                           warmup_steps=self.warmup_steps,
                                                           start_lr=0.0,
                                                           end_lr=warm_up_end_lr)
            else:
                decayed_lr = self._piecewise_decay()
        elif self.lr_policy.lower() == 'cosine':
            if self.warm_up:
                warm_up_end_lr = self.base_lr*0.5*(math.cos(self.warmup_epoch*math.pi/self.epoch_nums)+1)
                print('cosine warm_up_end_lr：', warm_up_end_lr)
                decayed_lr = fluid.layers.linear_lr_warmup(self._cosine_decay(),
                                                           warmup_steps=self.warmup_steps,
                                                           start_lr=0.0,
                                                           end_lr=warm_up_end_lr)
            else:
                decayed_lr = self._cosine_decay()
        else:
            raise Exception(
                "unsupport learning decay policy! only support poly,piecewise,cosine"
            )
        return decayed_lr


if __name__ == '__main__':
    epoch_nums = 200
    step_per_epoch = 180
    base_lr = 0.003
    warmup_epoch = 5   # 热身数
    lr_scheduler = Lr(lr_policy='poly', base_lr=base_lr, epoch_nums=epoch_nums, step_per_epoch=step_per_epoch,
                      warm_up=True, warmup_epoch=warmup_epoch, decay_epoch=[50])
    lr = lr_scheduler.get_lr()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    lr_list = []
    for epoch in range(epoch_nums):
        for i in range(step_per_epoch):
            x = exe.run(fluid.default_main_program(),
                        fetch_list=[lr])
            lr_list.append(x[0])
            # print(x[0])
    # 绘图
    from matplotlib import pyplot as plt
    plt.plot(range(epoch_nums*step_per_epoch), lr_list)
    plt.xlabel('step')
    plt.ylabel('lr')
    plt.show()

