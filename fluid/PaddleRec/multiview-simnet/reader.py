import random

class Dataset:
    def __init__(self):
        pass

class SyntheticDataset(Dataset):
    def __init__(self, sparse_feature_dim, 
                 query_slot_num,
                 title_slot_num):
        # ids are randomly generated
        self.ids_per_slot = 10
        self.sparse_feature_dim = sparse_feature_dim
        self.query_slot_num = query_slot_num
        self.title_slot_num = title_slot_num
        self.dataset_size = 10000

    def _reader_creator(self, is_train):
        def generate_ids(num, space):
            return [random.randint(0, space - 1) for i in range(num)]

        def reader():
            for i in range(self.dataset_size):
                query_slots = []
                pos_title_slots = []
                neg_title_slots = []
                for i in range(self.query_slot_num):
                    qslot = generate_ids(self.ids_per_slot, self.sparse_feature_dim)
                    query_slots.append(qslot)
                for i in range(self.title_slot_num):
                    pt_slot = generate_ids(self.ids_per_slot, self.sparse_feature_dim)
                    pos_title_slots.append(pt_slot)
                if is_train:
                    for i in range(self.title_slot_num):
                        nt_slot = generate_ids(self.ids_per_slot, self.sparse_feature_dim)
                        neg_title_slots.append(nt_slot)
                    yield query_slots + pos_title_slots + neg_title_slots
                else:
                    yield query_slots + pos_title_slots

        return reader

    def train(self):
        return self._reader_creator(True)

    def valid(self):
        return self._reader_creator(True)

    def test(self):
        return self._reader_creator(False)
    
