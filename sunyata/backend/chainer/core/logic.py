from chainer import functions as F
from chainer import Variable
import numpy as np

from ...base.core.logic import BaseLogicAPI


class ChainerLogicAPI(BaseLogicAPI):
    def __init__(self):
        BaseLogicAPI.__init__(self)

    def minimum(self, a, b):
        assert a.dtype == b.dtype
        if a.dtype.name.startswith('float'):
            x = F.minimum(a, b)
        else:
            x = Variable(np.minimum(a.data, b.data))
        return x

    def maximum(self, a, b):
        assert a.dtype == b.dtype
        if a.dtype.name.startswith('float'):
            x = F.maximum(a, b)
        else:
            x = Variable(np.maximum(a.data, b.data))
        return x

    def equal(self, a, b, dtype=None):
        data = a.data == b.data
        x = Variable(data)
        return self._cast_bool_output(a, x, dtype)

    def not_equal(self, a, b, dtype=None):
        data = a.data != b.data
        x = Variable(data)
        return self._cast_bool_output(a, x, dtype)

    def less(self, a, b, dtype=None):
        data = a.data < b.data
        x = Variable(data)
        return self._cast_bool_output(a, x, dtype)

    def less_equal(self, a, b, dtype=None):
        data = a.data <= b.data
        x = Variable(data)
        return self._cast_bool_output(a, x, dtype)

    def greater_equal(self, a, b, dtype=None):
        data = a.data >= b.data
        x = Variable(data)
        return self._cast_bool_output(a, x, dtype)

    def greater(self, a, b, dtype=None):
        data = a.data > b.data
        x = Variable(data)
        return self._cast_bool_output(a, x, dtype)
