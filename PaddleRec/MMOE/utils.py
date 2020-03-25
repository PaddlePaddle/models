import random
import pandas as pd
import numpy as np
import os
import paddle.fluid as fluid
import io
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
import warnings


##按行读取文件
def reader_creator(file_dir):
    def reader():
        files = os.listdir(file_dir)
        for fi in files:
            with io.open(
                    os.path.join(file_dir, fi), "r", encoding='utf-8') as f:
                for l in islice(f, 1, None):  ##忽略第一行
                    l = l.strip().split(',')
                    l = list(map(float, l))
                    label_income = []
                    label_marital = []
                    data = l[2:]
                    if int(l[1]) == 0:
                        label_income = [1, 0]
                    elif int(l[1]) == 1:
                        label_income = [0, 1]
                    if int(l[0]) == 0:
                        label_marital = [1, 0]
                    elif int(l[0]) == 1:
                        label_marital = [0, 1]
                    label_income = np.array(label_income)
                    label_marital = np.array(label_marital)
                    #label = np.array()
                    #label.append(label_income)
                    #label.append(label_marital)
                    yield data, label_income, label_marital

    return reader


##读取一个batch
def batch_reader(reader, batch_size):
    def batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if (len(b) == batch_size):
                yield b
                b = []
        #if len(b) != 0:
        #    yield b
        #
    return batch_reader


##准备数据          
def prepare_reader(data_path, batch_size):
    data_set = reader_creator(data_path)
    #random.shuffle(data_set)
    return batch_reader(data_set, batch_size)
