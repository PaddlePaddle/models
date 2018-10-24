"""
This Module provide pretrained word-embeddings 
"""

from __future__ import print_function 
import sys
import numpy as np
import time, datetime

def Glove840B_300D(filepath, keys=None):
    """
    input: the "glove.840B.300d.txt" file path
    return: a dict, key: word (unicode), value: a numpy array with shape [300]
    """
    if keys is not None: 
        assert(isinstance(keys, set))
    sys.stderr.write("loading word2vec from %s\n" % filepath)
    sys.stderr.write("please wait for a minute.\n")
    start = time.time()
    word2vec = {}

    with open(filepath, "r") as f:
        for line in f:
            info = line.strip().split()
            # TODO: test python3
            word = info[0].decode('utf-8')
            if (keys is not None) and (word not in keys):
                continue
            vector = info[1:]
            assert(len(vector) == 300)
            word2vec[word] = np.asarray(vector, dtype='float32')

    end = time.time()
    sys.stderr.write("Spent %s on loading word2vec.\n" % str(datetime.timedelta(seconds=end-start)))
    return word2vec
 
if __name__ == '__main__':
    embed_dict = Glove840B_300D("data/glove.840B.300d.txt")
