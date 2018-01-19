class Dataset:
    def _reader_creator(self, path, is_infer):
        def reader():
            with open(path, 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')
                    dense_feature = map(float, features[0].split(','))
                    sparse_feature = map(int, features[1].split(','))
                    if not is_infer:
                        label = [float(features[2])]
                        yield [dense_feature, sparse_feature
                               ] + sparse_feature + [label]
                    else:
                        yield [dense_feature, sparse_feature] + sparse_feature

        return reader

    def train(self, path):
        return self._reader_creator(path, False)

    def test(self, path):
        return self._reader_creator(path, False)

    def infer(self, path):
        return self._reader_creator(path, True)


feeding = {
    'dense_input': 0,
    'sparse_input': 1,
    'C1': 2,
    'C2': 3,
    'C3': 4,
    'C4': 5,
    'C5': 6,
    'C6': 7,
    'C7': 8,
    'C8': 9,
    'C9': 10,
    'C10': 11,
    'C11': 12,
    'C12': 13,
    'C13': 14,
    'C14': 15,
    'C15': 16,
    'C16': 17,
    'C17': 18,
    'C18': 19,
    'C19': 20,
    'C20': 21,
    'C21': 22,
    'C22': 23,
    'C23': 24,
    'C24': 25,
    'C25': 26,
    'C26': 27,
    'label': 28
}
