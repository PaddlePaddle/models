import random


class Dataset:
    def __init__(self):
        pass


class SyntheticDataset(Dataset):
    def __init__(self, sparse_feature_dim, user_slot_num, title_slot_num):
        # ids are randomly generated
        self.ids_per_slot = 10
        self.sparse_feature_dim = sparse_feature_dim
        self.user_slot_num = user_slot_num
        self.title_slot_num = title_slot_num
        self.dataset_size = 10000

    def _reader_creator(self, is_train):
        def generate_ids(num, space):
            return [random.randint(0, space - 1) for i in range(num)]

        def reader():
            for i in range(self.dataset_size):
                user_slots = []
                pos_title_slots = []
                neg_title_slots = []
                for i in range(self.user_slot_num):
                    uslot = generate_ids(self.ids_per_slot,
                                         self.sparse_feature_dim)
                    user_slots.append(uslot)
                for i in range(self.title_slot_num):
                    pt_slot = generate_ids(self.ids_per_slot,
                                           self.sparse_feature_dim)
                    pos_title_slots.append(pt_slot)
                for i in range(self.title_slot_num):
                    nt_slot = generate_ids(self.ids_per_slot,
                                           self.sparse_feature_dim)
                    neg_title_slots.append(nt_slot)
                yield user_slots + pos_title_slots + neg_title_slots

        return reader

    def train(self):
        return self._reader_creator(True)

    def valid(self):
        return self._reader_creator(True)

    def test(self):
        return self._reader_creator(False)


if __name__ == '__main__':
    dataset = SyntheticDataset(1000, 5, 5)
    reader = dataset.train()
    for _ in reader():
        print(_)
