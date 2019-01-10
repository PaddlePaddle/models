import sys
import os
import paddle


def parse_fields(fields):
    words_width = int(fields[0])
    words = fields[1:1 + words_width]
    label = fields[-1]

    return words, label


def imdb_data_feed_reader(data_dir, batch_size, buf_size):
    """ 
    Data feed reader for IMDB dataset.
    This data set has been converted from original format to a format suitable
    for AsyncExecutor
    See data.proto for data format
    """

    def reader():
        for file in os.listdir(data_dir):
            if file.endswith('.proto'):
                continue

            with open(os.path.join(data_dir, file), 'r') as f:
                for line in f:
                    fields = line.split(' ')
                    words, label = parse_fields(fields)
                    yield words, label

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            reader, buf_size=buf_size), batch_size=batch_size)
    return test_reader
