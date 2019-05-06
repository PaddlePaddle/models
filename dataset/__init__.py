"""
module to prepare data for detection model training

implementation notes:
- Dateset
    basic interface to accessing data samples in stream mode

- xxxSource (RoiDbSource)
    * subclass of 'Dataset'
    * load data from local files and other source data

- xxxOperator (DecodeImage)
    * subclass of 'BaseOperator' 
    * each op can transform a sample, eg: decode/resize/crop image
    * each op must obey basic rules defined in transform.operator.base

- Transformer
    * accept a 'xxxSource' and a list of 'xxxOperator' to provide a transformed 'Dataset'
    * naive implementation just pull sample from source and then transform it

"""

from .dataset import Dataset
from . import source
from . import transform

build_source = source.build

def build_dataset(config):
    """ build a transformed dataset by:
        1, loading data from 'config.source'
        2, transform sample using 'config.ops'
        3, accelerate it using multiple workers
    """
    sc_conf = config['source']
    op_conf = config['ops']
    worker_conf = config['worker_args']

    sc = source.build(sc_conf)
    return transform.transform(sc, \
        op_conf, worker_conf)


__all__ = ['Dataset', 'source', \
    'build_source', 'build_dataset']
