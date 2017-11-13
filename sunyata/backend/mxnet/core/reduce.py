import mxnet as mx

from ...base.core.reduce import BaseReduceAPI


class MXNetReduceAPI(BaseReduceAPI):
    def __init__(self):
        BaseReduceAPI.__init__(self)

    def argmin(self, x, axis=-1):
        return mx.nd.argmin(x, axis)

    def argmax(self, x, axis=-1):
        return mx.nd.argmax(x, axis)

    def _reduce(self, name, x, axis=None, keepdims=False):
        axis = mx.base._Null if axis is None else axis
        func = getattr(mx.nd, name)
        return func(x, axis, keepdims)

    def min(self, x, axis=None, keepdims=False):
        return self._reduce(x, axis, keepdims, 'min')

    def max(self, x, axis=None, keepdims=False):
        return self._reduce(x, axis, keepdims, 'max')

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return self._reduce('mean', x, axis, keepdims)

    def prod(self, x, axis=None, keepdims=False):
        return self._reduce('prod', x, axis, keepdims)
