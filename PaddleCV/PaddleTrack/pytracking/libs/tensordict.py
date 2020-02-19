from collections import OrderedDict


class TensorDict(OrderedDict):
    """Container mainly used for dicts of Variable."""

    def concat(self, other):
        """Concatenates two dicts without copying internal data."""
        return TensorDict(self, **other)

    def copy(self):
        return TensorDict(super(TensorDict, self).copy())

    def __getattr__(self, name):
        for n, e in self.items():
            if not hasattr(e, name):
                raise AttributeError('\'{}\' object has not attribute \'{}\''.
                                     format(type(e), name))

        def apply_attr(*args, **kwargs):
            return TensorDict({
                n: getattr(e, name)(*args, **kwargs) if hasattr(e, name) else e
                for n, e in self.items()
            })

        return apply_attr

    def attribute(self, attr: str, *args):
        return TensorDict({n: getattr(e, attr, *args) for n, e in self.items()})

    def apply(self, fn, *args, **kwargs):
        return TensorDict({n: fn(e, *args, **kwargs) for n, e in self.items()})

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorDict, list))
