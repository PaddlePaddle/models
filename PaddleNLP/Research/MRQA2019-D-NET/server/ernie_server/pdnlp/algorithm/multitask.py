#encoding=utf8

import os
import sys
import random
from copy import deepcopy as copy
import numpy as np
import paddle
import paddle.fluid as fluid
import multiprocessing

class Task:

    def __init__(
        self, 
        conf,
        name = "",
        is_training = False,
        _DataProcesser = None,
        shared_name = ""):
        
        self.conf = copy(conf)

        self.name = name
        self.shared_name = shared_name

        self.is_training = is_training
        self.DataProcesser = _DataProcesser

    def _create_reader(self):
        raise NotImplementedError("Task:_create_reader not implemented")

    def _create_model(self):
        raise NotImplementedError("Task:_create_model not implemented")

    def prepare(self, args):
        raise NotImplementedError("Task:prepare not implemented")

    def train_step(self, args):
        raise NotImplementedError("Task:train_step not implemented")

    def predict(self, args):
        raise NotImplementedError("Task:_predict not implemented")


class JointTask:

    def __init__(self):

        self.tasks = []

        #self.startup_exe = None
        #self.train_exe = None
       
        self.exe = None

        self.share_vars_from = None

        self.startup_prog = fluid.Program()

    def __add__(self, task):

        assert isinstance(task, Task)

        self.tasks.append(task)

        return self

    def prepare(self, args):

        if args.use_cuda:
            place = fluid.CUDAPlace(0)
            dev_count = fluid.core.get_cuda_device_count()
        else:
            place = fluid.CPUPlace()
            dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        #self.startup_exe = fluid.Executor(place)
        self.exe = fluid.Executor(place)

        for idx, task in enumerate(self.tasks):
            if idx == 0:
                print("for idx : %d" % idx)
                task.prepare(args, exe = self.exe)
                self.share_vars_from = task.compiled_train_prog
            else:
                print("for idx : %d" % idx)
                task.prepare(args, exe = self.exe, share_vars_from = self.share_vars_from)

    def train(self, args):

        joint_steps = []
        for i in xrange(0, len(self.tasks)):
            for _ in xrange(0, self.tasks[i].max_train_steps):
                joint_steps.append(i)

        self.tasks[0].train_step(args, exe = self.exe)

        random.shuffle(joint_steps)
        for next_task_id in joint_steps:
            self.tasks[next_task_id].train_step(args, exe = self.exe)


if __name__ == "__main__":

    basetask_a = Task(None)

    basetask_b = Task(None)

    joint_tasks = JointTask()

    joint_tasks += basetask_a

    print(joint_tasks.tasks)

    joint_tasks += basetask_b

    print(joint_tasks.tasks)

