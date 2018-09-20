import numpy as np
def recall_topk(fea, lab, k = 1):
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1)
    n = np.sqrt(np.sum(fea**2, 1)).reshape(-1, 1)
    fea = fea/n
    a = np.sum(fea ** 2, 1).reshape(-1, 1)
    b = a.T
    ab = np.dot(fea, fea.T)
    d = a + b - 2*ab
    d = d + np.eye(len(fea)) * 1e8
    sorted_index = np.argsort(d, 1)
    res = 0
    for i in range(len(fea)):
        pred = lab[sorted_index[i][0]]
        if lab[i] == pred:
            res += 1.0
    res = res/len(fea)
    return res

import subprocess
import os
def get_gpu_num():
    visibledevice = os.getenv('CUDA_VISIBLE_DEVICES')
    if visibledevice:
        devicenum = len(visibledevice.split(','))
    else:
        devicenum = subprocess.check_output(['nvidia-smi', '-L']).count('\n')
    return devicenum

import paddle as paddle
import paddle.fluid as fluid

def generate_index(batch_size, samples_each_class):
    a = np.arange(0, batch_size * batch_size)
    a = a.reshape(-1, batch_size)
    steps = batch_size // samples_each_class
    res = []
    for i in range(batch_size):
        step = i // samples_each_class
        start = step * samples_each_class
        end = (step + 1) * samples_each_class
        p = []
        n = []
        for j, k in enumerate(a[i]):
            if j >= start and j < end:
                if j == i:
                    p.insert(0, k)
                else:
                    p.append(k)
            else:
                n.append(k)
        comb = p + n
        res += comb
    res = np.array(res).astype(np.int32)
    return res

def calculate_order_dist_matrix(feature, batch_size, samples_each_class):
    assert(batch_size % samples_each_class == 0)
    feature = fluid.layers.reshape(feature, shape=[batch_size, -1])
    ab = fluid.layers.matmul(feature, feature, False, True)
    a2 = fluid.layers.square(feature)
    a2 = fluid.layers.reduce_sum(a2, dim = 1)
    d = fluid.layers.elementwise_add(-2*ab, a2, axis = 0)
    d = fluid.layers.elementwise_add(d, a2, axis = 1)
    d = fluid.layers.reshape(d, shape = [-1, 1])
    index = generate_index(batch_size, samples_each_class)
    index_var = fluid.layers.create_global_var(shape=[batch_size*batch_size], value=0, dtype='int32', persistable=True)
    index_var = fluid.layers.assign(index, index_var)
    d = fluid.layers.gather(d, index=index_var)
    d = fluid.layers.reshape(d, shape=[-1, batch_size])
    return d
    



