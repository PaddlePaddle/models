""" interface for accessing data samples in stream
"""
class Dataset(object):
    """ interface to access a stream of data samples
    """
    def __init__(self):
        self._epoch = -1

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def next(self):
        """ get next sample
        """
        raise NotImplementedError()

    def reset(self):
        """ reset to initial status and begins a new epoch
        """
        raise NotImplementedError()

    def size(self):
        """ get number of samples in this dataset
        """
        raise NotImplementedError()

    def drained(self):
        """ whether all sampled has been readed out for this epoch
        """
        raise NotImplementedError()

    def epoch_id(self):
        """ return epoch id for latest sample
        """
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return self._epoch

