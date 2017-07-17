from utils import logger, TaskMode, load_dnn_input_record, load_lr_input_record

feeding_index = {'dnn_input': 0, 'lr_input': 1, 'click': 2}


class Dataset(object):
    def __init__(self):
        self.mode = TaskMode.create_train()

    def train(self, path):
        '''
        Load trainset.
        '''
        logger.info("load trainset from %s" % path)
        self.mode = TaskMode.create_train()
        self.path = path
        return self._parse

    def test(self, path):
        '''
        Load testset.
        '''
        logger.info("load testset from %s" % path)
        self.path = path
        self.mode = TaskMode.create_test()
        return self._parse

    def infer(self, path):
        '''
        Load infer set.
        '''
        logger.info("load inferset from %s" % path)
        self.path = path
        self.mode = TaskMode.create_infer()
        return self._parse

    def _parse(self):
        '''
        Parse dataset.
        '''
        with open(self.path) as f:
            for line_id, line in enumerate(f):
                fs = line.strip().split('\t')
                dnn_input = load_dnn_input_record(fs[0])
                lr_input = load_lr_input_record(fs[1])
                if not self.mode.is_infer():
                    click = [int(fs[2])]
                    yield dnn_input, lr_input, click
                else:
                    yield dnn_input, lr_input


def load_data_meta(path):
    '''
    load data meta info from path, return (dnn_input_dim, lr_input_dim)
    '''
    with open(path) as f:
        lines = f.read().split('\n')
        err_info = "wrong meta format"
        assert len(lines) == 2, err_info
        assert 'dnn_input_dim:' in lines[0] and 'lr_input_dim:' in lines[
            1], err_info
        res = map(int, [_.split(':')[1] for _ in lines])
        logger.info('dnn input dim: %d' % res[0])
        logger.info('lr input dim: %d' % res[1])
        return res
