from utils import logger, TaskMode, load_dnn_input_record, load_lr_input_record

feeding_index = {'dnn_input': 0, 'lr_input': 1, 'click': 2}


class Dataset(object):
    def train(self, path):
        '''
        Load trainset.
        '''
        logger.info("load trainset from %s" % path)
        mode = TaskMode.create_train()
        return self._parse_creator(path, mode)

    def test(self, path):
        '''
        Load testset.
        '''
        logger.info("load testset from %s" % path)
        mode = TaskMode.create_test()
        return self._parse_creator(path, mode)

    def infer(self, path):
        '''
        Load infer set.
        '''
        logger.info("load inferset from %s" % path)
        mode = TaskMode.create_infer()
        return self._parse_creator(path, mode)

    def _parse_creator(self, path, mode):
        '''
        Parse dataset.
        '''

        def _parse():
            with open(path) as f:
                for line_id, line in enumerate(f):
                    fs = line.strip().split('\t')
                    dnn_input = load_dnn_input_record(fs[0])
                    lr_input = load_lr_input_record(fs[1])
                    if not mode.is_infer():
                        click = [int(fs[2])]
                        yield dnn_input, lr_input, click
                    else:
                        yield dnn_input, lr_input

        return _parse


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
