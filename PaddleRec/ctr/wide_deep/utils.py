import numpy as np 
import os
import paddle.fluid as fluid

class Dataset(object):

    def _reader_creator(self, file):
        def reader():
            with open(file, 'r') as f:
                for i,line in enumerate(f):
                    if i == 0:
                        continue
                    line = line.strip().split(',')
                    features = list(map(float, line))
                    wide_feat = features[0:8]
                    deep_feat = features[8:58+8]
                    label = features[-1]
                    output = []
                    output.append(wide_feat)
                    output.append(deep_feat)
                    output.append([label])
                    
                    yield output

        return reader

    def train(self, file):
        return self._reader_creator(file)

    def test(self, file):
        return self._reader_creator(file)