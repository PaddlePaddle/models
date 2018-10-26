class Dataset:
    def _reader_creator(self, file_list, is_infer):
        def reader():
            for file in file_list:
                with open(file, 'r') as f:
                    for line in f:
                        features = line.rstrip('\n').split('\t')
                        dense_feature = map(float, features[0].split(','))
                        sparse_feature = map(lambda x: [int(x)], features[1].split(','))
                        if not is_infer:
                            label = [float(features[2])]
                            yield [dense_feature
                                   ] + sparse_feature + [label]
                        else:
                            yield [dense_feature] + sparse_feature

        return reader

    def train(self, file_list):
        return self._reader_creator(file_list, False)

    def test(self, file_list):
        return self._reader_creator(file_list, False)

    def infer(self, file_list):
        return self._reader_creator(file_list, True)
