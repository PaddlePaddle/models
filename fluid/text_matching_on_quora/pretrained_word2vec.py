"""
This Module provide pretrained word-embeddings 
"""

from __future__ import print_function 
import numpy as np

def Glove840B_300D(filepath="data/glove.840B.300d.txt"):
    """
    input: the "glove.840B.300d.txt" file path
    return: a dict, key: word (unicode), value: a numpy array with shape [300]
    """
    print("loading word2vec from ", filepath)
    word2vec = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip().split()
            word, vector = info[0], info[1:]
            assert(len(vector) == 300)
            #TODO: test python3
            word2vec[word.decode('utf-8')] = np.asarray(vector, dtype='float32')
    return word2vec
 
if __name__ == '__main__':
    embed_dict = Glove840B_300D("data/glove.840B.300d.txt")
