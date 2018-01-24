import numpy as np
DATA_SHAPE = [1, 512, 512]


def _read_creater(num_sample=1024, num_class=20, min_seq_len=1, max_seq_len=10):
    def reader():
        for i in range(num_sample):
            sequence_len = np.random.randint(min_seq_len, max_seq_len)
            x = np.random.uniform(0.1, 1, DATA_SHAPE).astype("float32")
            y = np.random.randint(0, num_class + 1,
                                  [sequence_len]).astype("int32")
            yield x, y

    return reader


def train(num_sample=16):
    return _read_creater(num_sample=num_sample)


def test(num_sample=16):
    return _read_creater(num_sample=num_sample)


def data_shape():
    return DATA_SHAPE
