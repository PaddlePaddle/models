"""
    transform samples in 'source' using 'mapper'
"""

from ..dataset import Dataset
class Transformer(Dataset):
    """ simple transformer without any workers to accelerate the processing
elerate                                                           
    """
    def __init__(self, source, mapper, worker_args=None):
        self._source = source
        self._mapper = mapper

    def next(self):
        sample = self._source.next()
        return self._mapper(sample)

    def reset(self):
        self._source.reset()

    def drained(self):
        return self._source.drained()

    def epoch_id(self):
        return self._source.epoch_id()

