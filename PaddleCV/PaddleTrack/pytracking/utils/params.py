from pytracking.libs import TensorList
import random


class TrackerParams:
    """Class for tracker parameters."""
    def free_memory(self):
        for a in dir(self):
            if not a.startswith('__') and hasattr(getattr(self, a), 'free_memory'):
                getattr(self, a).free_memory()


class FeatureParams:
    """Class for feature specific parameters"""
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError

        for name, val in kwargs.items():
            if isinstance(val, list):
                setattr(self, name, TensorList(val))
            else:
                setattr(self, name, val)


def Choice(*args):
    """Can be used to sample random parameter values."""
    return random.choice(args)
