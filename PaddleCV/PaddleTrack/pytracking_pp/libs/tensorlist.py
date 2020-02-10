import functools
import numpy as np
from paddle.fluid import layers

from pytracking_pp.libs.paddle_utils import clone as clone_fn
from pytracking_pp.libs.paddle_utils import detach as detach_fn
from pytracking_pp.libs.paddle_utils import PTensor


def matmul(a, b):
    if isinstance(a, PTensor) or isinstance(b, PTensor):
        return layers.matmul(a, b)
    else:
        return np.matmul(a, b)


class TensorList(list):
    """Container mainly used for lists of paddle tensors. Extends lists with paddle functionality."""

    def __init__(self, list_of_tensors=list()):
        super(TensorList, self).__init__(list_of_tensors)

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(TensorList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return TensorList([super(TensorList, self).__getitem__(i) for i in item])
        else:
            return TensorList(super(TensorList, self).__getitem__(item))

    def __add__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 + e2 for e1, e2 in zip(self, other)])
        return TensorList([e + other for e in self])

    def __radd__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 + e1 for e1, e2 in zip(self, other)])
        return TensorList([other + e for e in self])

    def __iadd__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] += e2
        else:
            for i in range(len(self)):
                self[i] += other
        return self

    def __sub__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 - e2 for e1, e2 in zip(self, other)])
        return TensorList([e - other for e in self])

    def __rsub__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 - e1 for e1, e2 in zip(self, other)])
        return TensorList([other - e for e in self])

    def __isub__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] -= e2
        else:
            for i in range(len(self)):
                self[i] -= other
        return self

    def __mul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 * e2 for e1, e2 in zip(self, other)])
        return TensorList([e * other for e in self])

    def __rmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 * e1 for e1, e2 in zip(self, other)])
        return TensorList([other * e for e in self])

    def __imul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] *= e2
        else:
            for i in range(len(self)):
                self[i] *= other
        return self

    def __truediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 / e2 for e1, e2 in zip(self, other)])
        return TensorList([e / other for e in self])

    def __rtruediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 / e1 for e1, e2 in zip(self, other)])
        return TensorList([other / e for e in self])

    def __itruediv__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] /= e2
        else:
            for i in range(len(self)):
                self[i] /= other
        return self

    def __matmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([matmul(e1, e2) for e1, e2 in zip(self, other)])
        return TensorList([matmul(e, other) for e in self])

    def __rmatmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([matmul(e2, e1) for e1, e2 in zip(self, other)])
        return TensorList([matmul(other, e) for e in self])

    def __imatmul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] = matmul(self[i], e2)
        else:
            for i in range(len(self)):
                self[i] = matmul(self[i], other)
        return self

    def __mod__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 % e2 for e1, e2 in zip(self, other)])
        return TensorList([e % other for e in self])

    def __rmod__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 % e1 for e1, e2 in zip(self, other)])
        return TensorList([other % e for e in self])

    def __pos__(self):
        return TensorList([+e for e in self])

    def __neg__(self):
        return TensorList([-e for e in self])

    def __le__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 <= e2 for e1, e2 in zip(self, other)])
        return TensorList([e <= other for e in self])

    def __ge__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 >= e2 for e1, e2 in zip(self, other)])
        return TensorList([e >= other for e in self])

    def view(self, *args):
        def reshape(x):
            if isinstance(x, PTensor):
                return layers.reshape(x, args)
            else:
                return np.reshape(x, args)

        return self.apply(reshape)

    def clone(self):
        def _clone(x):
            if isinstance(x, PTensor):
                return clone_fn(x)
            else:
                return x.copy()

        return self.apply(_clone)

    def detach(self):
        return self.apply(detach_fn)

    def sqrt(self):
        def _sqrt(x):
            if isinstance(x, PTensor):
                return layers.sqrt(x)
            else:
                return np.sqrt(x)

        return self.apply(_sqrt)

    def abs(self):
        def _abs(x):
            if isinstance(x, PTensor):
                return layers.abs(x)
            else:
                return np.abs(x)

        return self.apply(_abs)

    def size(self, axis=None):
        def get_size(x):
            if axis is None:
                return x.shape
            else:
                return x.shape[axis]

        return self.apply(get_size)

    def concat(self, other):
        return TensorList(super(TensorList, self).__add__(other))

    def copy(self):
        return TensorList(super(TensorList, self).copy())

    def unroll(self):
        if not any(isinstance(t, TensorList) for t in self):
            return self

        new_list = TensorList()
        for t in self:
            if isinstance(t, TensorList):
                new_list.extend(t.unroll())
            else:
                new_list.append(t)
        return new_list

    def attribute(self, attr: str, *args):
        return TensorList([getattr(e, attr, *args) for e in self])

    def apply(self, fn):
        return TensorList([fn(e) for e in self])

    def __getattr__(self, name):
        for e in self:
            if not hasattr(e, name):
                raise AttributeError('\'{}\' object has not attribute \'{}\''.format(type(e), name))

        def apply_attr(*args, **kwargs):
            return TensorList([getattr(e, name)(*args, **kwargs) for e in self])

        return apply_attr

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorList, list))


def tensor_operation(op):
    def islist(a):
        return isinstance(a, TensorList)

    @functools.wraps(op)
    def oplist(*args, **kwargs):
        if len(args) == 0:
            raise ValueError('Must be at least one argument without keyword (i.e. operand).')

        if len(args) == 1:
            if islist(args[0]):
                return TensorList([op(a, **kwargs) for a in args[0]])
        else:
            # Multiple operands, assume max two
            if islist(args[0]) and islist(args[1]):
                return TensorList([op(a, b, *args[2:], **kwargs) for a, b in zip(*args[:2])])
            if islist(args[0]):
                return TensorList([op(a, *args[1:], **kwargs) for a in args[0]])
            if islist(args[1]):
                return TensorList([op(args[0], b, *args[2:], **kwargs) for b in args[1]])

        # None of the operands are lists
        return op(*args, **kwargs)

    return oplist
