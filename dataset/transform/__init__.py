"""
transform a dataset to another one
"""

from . import transformer
from . import operator

def transform(source, ops_conf, worker_args=None):
    """ transform data in 'source' using a mapper defined by 'ops_conf'

    Args:
        @source (instance of Dataset): input data sample
        @ops_conf (list of op configs): used to build a mapper which accept a sample and return a transformed sample

    Returns:
        instance of 'Dataset'
    """
    mapper = operator.build(ops_conf)
    return transformer.Transformer(source, \
        mapper, worker_args)


__all__ = ['transformer']
