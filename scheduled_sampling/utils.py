import math
import numpy as np


class RandomScheduleGenerator:
    """
    The random sampling rate for scheduled sampling algoithm, which uses decayed
    sampling rate.
    """

    def __init__(self, schedule_type, a, b):
        """
        schduled_type: is the type of the decay. It supports constant, linear,
        exponential, and inverse_sigmoid right now.
        a: parameter of the decay (MUST BE DOUBLE)
        b: parameter of the decay (MUST BE DOUBLE)
        """
        self.schedule_type = schedule_type
        self.a = a
        self.b = b
        self.data_processed_ = 0
        self.schedule_computers = {
            "constant": lambda a, b, d: a,
            "linear": lambda a, b, d: max(a, 1 - d / b),
            "exponential": lambda a, b, d: pow(a, d / b),
            "inverse_sigmoid": lambda a, b, d: b / (b + math.exp(d * a / b)),
        }
        assert (self.schedule_type in self.schedule_computers)
        self.schedule_computer = self.schedule_computers[self.schedule_type]

    def getScheduleRate(self):
        """
        Get the schedule sampling rate. Usually not needed to be
        called by the users.
        """
        return self.schedule_computer(self.a, self.b, self.data_processed_)

    def processBatch(self, batch_size):
        """
        Get a batch_size of sampled indexes. These indexes can be passed to a
        MultiplexLayer to select from the grouth truth and generated samples
        from the last time step.
        """
        rate = self.getScheduleRate()
        numbers = np.random.rand(batch_size)
        indexes = (numbers >= rate).astype('int32').tolist()
        self.data_processed_ += batch_size
        return indexes
