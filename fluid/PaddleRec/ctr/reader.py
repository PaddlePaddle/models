class Dataset:
    def __init__(self):
        pass


class CriteoDataset(Dataset):
    def __init__(self, sparse_feature_dim, fix_id_range=True, extend_id_range=False):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.hash_dim_ = sparse_feature_dim
        self.fix_id_range_ = fix_id_range
        self.extend_id_range_ = extend_id_range
        # here, training data are lines with line_index < train_idx_
        self.train_idx_ = 41256555
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)

    def _reader_creator(self, file_list, is_train, trainer_num, trainer_id):
        def reader():
            for file in file_list:
                with open(file, 'r') as f:
                    line_idx = 0
                    for line in f:
                        line_idx += 1
                        if is_train and line_idx > self.train_idx_:
                            break
                        elif not is_train and line_idx <= self.train_idx_:
                            continue
                        if line_idx % trainer_num != trainer_id:
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
                            feature_id = hash("%d_%s" % (idx, features[idx]))
                            if self.fix_id_range_:
                                feature_id = feature_id % self.hash_dim_
                            sparse_feature.append([feature_id])
                        if self.extend_id_range_:
                            for i in range(len(self.categorical_range_)):
                                for j in range(i + 1, len(self.categorical_range_)):
                                    idx1 = self.categorical_range_[i]
                                    idx2 = self.categorical_range_[j]
                                    feature_id = hash("%d_%s_%d_%s" % (idx1, features[idx1], idx2, features[idx2]))
                                    if self.fix_id_range_:
                                        feature_id = feature_id % self.hash_dim_
                                    sparse_feature.append([feature_id])

                        label = [int(features[0])]
                        yield [dense_feature] + sparse_feature + [label]
                        
        return reader

    def train(self, file_list, trainer_num, trainer_id):
        return self._reader_creator(file_list, True, trainer_num, trainer_id)

    def test(self, file_list):
        return self._reader_creator(file_list, False, -1)

    def infer(self, file_list):
        return self._reader_creator(file_list, False, -1)
