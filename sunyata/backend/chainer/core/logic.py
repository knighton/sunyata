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

    def _compare(self, a, b, func_name):
        assert a.dtype == b.dtype
        func = getattr(a.data, func_name)
        data = func(b.data).astype(a.dtype)
        return Variable(data)

    def equal(self, a, b):
        return self._compare(a, b, '__eq__')

    def not_equal(self, a, b):
        return self._compare(a, b, '__ne__')

    def less(self, a, b):
        return self._compare(a, b, '__lt__')

    def less_equal(self, a, b):
        return self._compare(a, b, '__le__')

    def greater_equal(self, a, b):
        return self._compare(a, b, '__ge__')

    def greater(self, a, b):
        return self._compare(a, b, '__gt__')
