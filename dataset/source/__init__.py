import copy
from .datasource import RoiDbSource

def build(config):
    """ build dataset from source data, 
        default source type is 'RoiDbSource'
    """
    args = copy.deepcopy(config)
    if 'type' in config:
        source_type = config['type']
        del args['type']
    else:
        source_type = 'RoiDbSource'

    if source_type == 'RoiDbSource':
        return RoiDbSource(**args)
    else:
        raise ValueError('not supported source type[%s]' % (source_type))

