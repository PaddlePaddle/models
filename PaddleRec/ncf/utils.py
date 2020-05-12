import numpy as np 
import os
import paddle.fluid as fluid

class CriteoDataset(object):

    def _reader_creator(self, file, is_train):
        def reader():
            with open(file, 'r') as f:
                for i,line in enumerate(f):
                    line = line.strip().split(',')
                    features = list(map(int, line))
                    
                    output = []
                    output.append([features[0]])
                    output.append([features[1]])
                    if is_train:
                        output.append([features[2]])
                    
                    yield output

        return reader

    def train(self, file, is_train):
        return self._reader_creator(file, is_train)

    def test(self, file, is_train):
        return self._reader_creator(file, is_train)
        
def input_data(is_train):
    user_input = fluid.data(name="user_input", shape=[-1, 1], dtype="int64", lod_level=0)
    item_input = fluid.data(name="item_input", shape=[-1, 1], dtype="int64", lod_level=0)
    label = fluid.data(name="label", shape=[-1, 1], dtype="int64", lod_level=0)
    if is_train:
        inputs = [user_input] + [item_input] + [label]
    else:
        inputs = [user_input] + [item_input]
    
    return inputs