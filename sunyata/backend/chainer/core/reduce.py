from chainer import functions as F
from chainer import Variable
import numpy as np

from ...base.core.reduce import BaseReduceAPI


class ChainerReduceAPI(BaseReduceAPI):
    def argmin(self, x, axis=-1):
        return F.argmin(x, axis)

    def argmax(self, x, axis=-1):
        return F.argmax(x, axis)

    def _reduce(self, func_name, x, axis=None, keepdims=False):
        if isinstance(axis, list):
            axis = tuple(axis)
        dtype = self.dtype_of(x)
        if dtype.startswith('float'):
            x = getattr(F, func_name)(x, axis, keepdims)
        else:
            data = getattr(np, func_name)(x.data, axis, keepdims)
            x = Variable(data)
        if not x.ndim:
            x = self.expand_dims(x, 0)
        return x

    def min(self, x, axis=None, keepdims=False):
        return self._reduce('min', x, axis, keepdims)

    def max(self, x, axis=None, keepdims=False):
        return self._reduce('max', x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)

    def prod(self, x, axis=None, keepdims=False):
        return self._reduce('prod', x, axis, keepdims)
