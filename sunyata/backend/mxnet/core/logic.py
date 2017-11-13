import mxnet as mx

from ...base.core.logic import BaseLogicAPI


class MXNetLogicAPI(BaseLogicAPI):
    def __init__(self):
        BaseLogicAPI.__init__(self)

    def minimum(self, a, b):
        return mx.nd.broadcast_minimum(a, b)

    def maximum(self, a, b):
        return mx.nd.broadcast_maximum(a, b)

    def equal(self, a, b, dtype=None):
        x = mx.nd.broadcast_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def not_equal(self, a, b, dtype=None):
        x = mx.nd.broadcast_not_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def less(self, a, b, dtype=None):
        x = mx.nd.broadcast_less(a, b)
        return self._cast_bool_output(a, x, dtype)

    def less_equal(self, a, b, dtype=None):
        x = mx.nd.broadcast_less_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def greater_equal(self, a, b, dtype=None):
        x = mx.nd.broadcast_greater_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def greater(self, a, b, dtype=None):
        x = mx.nd.broadcast_greater(a, b)
        return self._cast_bool_output(a, x, dtype)
