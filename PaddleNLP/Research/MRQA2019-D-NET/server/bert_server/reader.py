#encoding=utf8
import os
import sys
import random
import numpy as np
import paddle
import paddle.fluid as fluid
from pdnlp.toolkit.placeholder import Placeholder


def repeat(reader):
    """Repeat a generator forever"""
    generator = reader()
    while True:
        try:
            yield next(generator)
        except StopIteration:
            generator = reader()
            yield next(generator)


def create_joint_generator(input_shape, generators, is_multi_task=True):

    def empty_output(input_shape, batch_size=1):
        results = []
        for i in range(len(input_shape)):
            if input_shape[i][1] == 'int32':
                dtype = np.int32
            if input_shape[i][1] == 'int64':
                dtype = np.int64
            if input_shape[i][1] == 'float32':
                dtype = np.float32
            if input_shape[i][1] == 'float64':
                dtype = np.float64
            shape = input_shape[i][0]
            shape[0] = batch_size
            pad_tensor = np.zeros(shape=shape, dtype=dtype)
            results.append(pad_tensor)
        return results

    def wrapper(): 
        """wrapper data"""
        generators_inst = [repeat(gen[0]) for gen in generators]

        generators_ratio = [gen[1] for gen in generators]
        weights = [ratio/sum(generators_ratio) for ratio in generators_ratio]
        run_task_id = range(len(generators))
        while True:
            idx = np.random.choice(run_task_id, p=weights)
            gen_results = next(generators_inst[idx])
            if not gen_results:
                break
            batch_size = gen_results[0].shape[0]
            results = empty_output(input_shape, batch_size)

            task_id_tensor = np.array([[idx]]).astype("int64")
            results[0] = task_id_tensor
            for i in range(4):
                results[i+1] = gen_results[i]
            if idx == 0:
                # mrc batch
                results[5] = gen_results[4]
                results[6] = gen_results[5]
            elif idx == 1:
                # mlm batch
                results[7] = gen_results[4]
                results[8] = gen_results[5]
            elif idx == 2:
                # MNLI batch
                results[9] = gen_results[4]
            else:
                raise RuntimeError('Invalid task ID - {}'.format(idx))
            # idx stands for the task index
            yield results

    return wrapper


def create_reader(reader_name, input_shape, is_multi_task, *gens):
    """
    build reader for multi_task_learning
    """
    placeholder = Placeholder(input_shape)
    pyreader, model_inputs = placeholder.build(capacity=100, reader_name=reader_name)
    joint_generator = create_joint_generator(input_shape, gens[0], is_multi_task=is_multi_task)

    return joint_generator, pyreader, model_inputs

