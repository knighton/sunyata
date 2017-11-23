from chainer import functions as F
from chainer import Variable
import numpy as np

from ...base.core.reduce import BaseReduceAPI


class ChainerReduceAPI(BaseReduceAPI):
    def argmin(self, x, axis=-1):
        return F.argmin(x, axis)

    def argmax(self, x, axis=-1):
        return F.argmax(x, axis)

    def _reduce_variable(self, func_name, x, axis, keepdims):
        axis = tuple(axis) if isinstance(axis, list) else axis
        return getattr(F, func_name)(x, axis, keepdims)

    def _reduce_numpy(self, func_name, x, axis, keepdims):
        axis = tuple(axis) if isinstance(x, list) else axis
        return getattr(np, func_name)(x, axis, keepdims=keepdims)

    def _reduce(self, func_name, x, axis=None, keepdims=False):
        if isinstance(x, Variable):
            if x.data.dtype.name.startswith('float'):
                x = self._reduce_variable(func_name, x, axis, keepdims)
            else:
                data = self._reduce_numpy(func_name, x.data, axis, keepdims)
                x = Variable(data)
        else:
            x = self._reduce_numpy(func_name, x, axis, keepdims)
        return x

    def min(self, x, axis=None, keepdims=False):
        return self._reduce('min', x, axis, keepdims)

    def max(self, x, axis=None, keepdims=False):
        return self._reduce('max', x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)

    def prod(self, x, axis=None, keepdims=False):
        return self._reduce('prod', x, axis, keepdims)

    def moments(self, x, axis=None, keepdims=False):
        shift = self.mean(x, axis, True)
        x, shift = F.broadcast(x, shift)
        shifted = x - shift
        shifted_mean = self.mean(shift, axis, True)
        var_mean = self.mean(self.square(shifted), axis, True)
        var = var_mean - self.square(shifted_mean)
        shifted_mean, shift = F.broadcast(shifted_mean, shift)
        mean = shifted_mean + shift
        return mean, var
