import sys
import csv
import cPickle
import numpy as np

from utils import logger, TaskMode
'''
The fields of the dataset are:

    0. id: ad identifier
    1. click: 0/1 for non-click/click
    2. hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
    3. C1 -- anonymized categorical variable
    4. banner_pos
    5. site_id
    6. site_domain
    7. site_category
    8. app_id
    9. app_domain
    10. app_category
    11. device_id
    12. device_ip
    13. device_model
    14. device_type
    15. device_conn_type
    16. C14-C21 -- anonymized categorical variables

We will treat following fields as categorical features:

    - C1
    - banner_pos
    - site_category
    - app_category
    - device_type
    - device_conn_type

and some other features as id features:

    - id
    - site_id
    - app_id
    - device_id

The `hour` field will be treated as a continuous feature and will be transformed
to one-hot representation which has 24 bits.
'''

feature_dims = {}

categorial_features = ('C1 banner_pos site_category app_category ' +
                       'device_type device_conn_type').split()

id_features = 'id site_id app_id device_id _device_id_cross_site_id'.split()


def get_all_field_names(mode=0):
    '''
    @mode: int
        0 for train, 1 for test
    @return: list of str
    '''
    return categorial_features + ['hour'] + id_features + ['click'] \
        if mode == 0 else []


class CategoryFeatureGenerator(object):
    '''
    Generator category features.

    Register all records by calling `register` first, then call `gen` to generate
    one-hot representation for a record.
    '''

    def __init__(self):
        self.dic = {'unk': 0}
        self.counter = 1

    def register(self, key):
        '''
        Register record.
        '''
        if key not in self.dic:
            self.dic[key] = self.counter
            self.counter += 1

    def size(self):
        return len(self.dic)

    def gen(self, key):
        '''
        Generate one-hot representation for a record.
        '''
        if key not in self.dic:
            res = self.dic['unk']
        else:
            res = self.dic[key]
        return [res]

    def __repr__(self):
        return '<CategoryFeatureGenerator %d>' % len(self.dic)


class IDfeatureGenerator(object):
    def __init__(self, max_dim, cross_fea0=None, cross_fea1=None):
        '''
        @max_dim: int
            Size of the id elements' space
        '''
        self.max_dim = max_dim
        self.cross_fea0 = cross_fea0
        self.cross_fea1 = cross_fea1

    def gen(self, key):
        '''
        Generate one-hot representation for records
        '''
        return [hash(key) % self.max_dim]

    def gen_cross_fea(self, fea1, fea2):
        key = str(fea1) + str(fea2)
        return self.gen(key)

    def size(self):
        return self.max_dim


class ContinuousFeatureGenerator(object):
    def __init__(self, n_intervals):
        self.min = sys.maxint
        self.max = sys.minint
        self.n_intervals = n_intervals

    def register(self, val):
        self.min = min(self.minint, val)
        self.max = max(self.maxint, val)

    def gen(self, val):
        self.len_part = (self.max - self.min) / self.n_intervals
        return (val - self.min) / self.len_part


# init all feature generators
fields = {}
for key in categorial_features:
    fields[key] = CategoryFeatureGenerator()
for key in id_features:
    # for cross features
    if 'cross' in key:
        feas = key[1:].split('_cross_')
        fields[key] = IDfeatureGenerator(10000000, *feas)
    # for normal ID features
    else:
        fields[key] = IDfeatureGenerator(10000)

# used as feed_dict in PaddlePaddle
field_index = dict((key, id)
                   for id, key in enumerate(['dnn_input', 'lr_input', 'click']))


def detect_dataset(path, topn, meta_out_path, id_fea_space=10000):
    '''
    Parse the first `topn` records to collect meta information of this dataset.

    NOTE the records should be randomly shuffled first.
    '''
    # create categorical statis objects.
    logger.warning('detecting dataset')

    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_id, row in enumerate(reader):
            if row_id > topn:
                break

            for key in categorial_features:
                fields[key].register(row[key])

    for key, item in fields.items():
        feature_dims[key] = item.size()

    feature_dims['hour'] = 24
    feature_dims['click'] = 1

    feature_dims['dnn_input'] = np.sum(
        feature_dims[key] for key in categorial_features + ['hour']) + 1
    feature_dims['lr_input'] = np.sum(feature_dims[key]
                                      for key in id_features) + 1
    logger.warning("dump dataset's meta info to %s" % meta_out_path)
    cPickle.dump([feature_dims, fields], open(meta_out_path, 'wb'))

    return feature_dims


def load_data_meta(meta_path):
    '''
    Load dataset's meta infomation.
    '''
    feature_dims, fields = cPickle.load(open(meta_path, 'rb'))
    return feature_dims, fields


def concat_sparse_vectors(inputs, dims):
    '''
    Concaterate more than one sparse vectors into one.

    @inputs: list
        list of sparse vector
    @dims: list of int
        dimention of each sparse vector
    '''
    res = []
    assert len(inputs) == len(dims)
    start = 0
    for no, vec in enumerate(inputs):
        for v in vec:
            res.append(v + start)
        start += dims[no]
    return res


class AvazuDataset(object):
    '''
    Load AVAZU dataset as train set.
    '''

    def __init__(self,
                 train_path,
                 n_records_as_test=-1,
                 fields=None,
                 feature_dims=None):
        self.train_path = train_path
        self.n_records_as_test = n_records_as_test
        self.fields = fields
        # default is train mode.
        self.mode = TaskMode.create_train()

        self.categorial_dims = [
            feature_dims[key] for key in categorial_features + ['hour']
        ]
        self.id_dims = [feature_dims[key] for key in id_features]

    def train(self):
        '''
        Load trainset.
        '''
        logger.info("load trainset from %s" % self.train_path)
        self.mode = TaskMode.create_train()
        with open(self.train_path) as f:
            reader = csv.DictReader(f)

            for row_id, row in enumerate(reader):
                # skip top n lines
                if self.n_records_as_test > 0 and row_id < self.n_records_as_test:
                    continue

                rcd = self._parse_record(row)
                if rcd:
                    yield rcd

    def test(self):
        '''
        Load testset.
        '''
        logger.info("load testset from %s" % self.train_path)
        self.mode = TaskMode.create_test()
        with open(self.train_path) as f:
            reader = csv.DictReader(f)

            for row_id, row in enumerate(reader):
                # skip top n lines
                if self.n_records_as_test > 0 and row_id > self.n_records_as_test:
                    break

                rcd = self._parse_record(row)
                if rcd:
                    yield rcd

    def infer(self):
        '''
        Load inferset.
        '''
        logger.info("load inferset from %s" % self.train_path)
        self.mode = TaskMode.create_infer()
        with open(self.train_path) as f:
            reader = csv.DictReader(f)

            for row_id, row in enumerate(reader):
                rcd = self._parse_record(row)
                if rcd:
                    yield rcd

    def _parse_record(self, row):
        '''
        Parse a CSV row and get a record.
        '''
        record = []
        for key in categorial_features:
            record.append(self.fields[key].gen(row[key]))
        record.append([int(row['hour'][-2:])])
        dense_input = concat_sparse_vectors(record, self.categorial_dims)

        record = []
        for key in id_features:
            if 'cross' not in key:
                record.append(self.fields[key].gen(row[key]))
            else:
                fea0 = self.fields[key].cross_fea0
                fea1 = self.fields[key].cross_fea1
                record.append(
                    self.fields[key].gen_cross_fea(row[fea0], row[fea1]))

        sparse_input = concat_sparse_vectors(record, self.id_dims)

        record = [dense_input, sparse_input]

        if not self.mode.is_infer():
            record.append(list((int(row['click']), )))
        return record


if __name__ == '__main__':
    path = 'train.txt'
    print detect_dataset(path, 400000)

    filereader = AvazuDataset(path)
    for no, rcd in enumerate(filereader.train()):
        print no, rcd
        if no > 1000: break
