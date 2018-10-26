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

class CriteoDataset:
    def __init__(self, sparse_feature_dim):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.hash_dim_ = sparse_feature_dim
        self.train_idx_ = 41256555
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)

    def _reader_creator(self, file_list, is_train, trainer_id):
        def reader():
            for file in file_list:
                with open(file, 'r') as f:
                    line_idx = 0
                    for line in f:
                        line_idx += 1
                        if is_train and line_idx > self.train_idx_:
                            continue
                        elif not is_train and line_idx <= self.train_idx_:
                            continue
                        if trainer_id > 0 and line_idx % trainer_id != 0:
                            continue
                        features = line.rstrip('\n').split('\t')
                        dense_feature = []
                        sparse_feature = []
                        for idx in self.continuous_range_:
                            if features[idx] == '':
                                dense_feature.append(0.0)
                            else:
                                dense_feature.append((float(features[idx]) - self.cont_min_[idx - 1]) / self.cont_diff_[idx - 1])
                        for idx in self.categorical_range_:
                            sparse_feature.append([hash("%d_%s" % (idx, features[idx])) % self.hash_dim_])

                        label = [int(features[0])]
                        yield [dense_feature] + sparse_feature + [label]
                        
        return reader

    def train(self, file_list, trainer_id):
        return self._reader_creator(file_list, True, trainer_id)

    def test(self, file_list):
        return self._reader_creator(file_list, False, -1)

    def infer(self, file_list):
        return self._reader_creator(file_list, False, -1)
