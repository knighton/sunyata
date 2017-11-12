from ..base import APIBase


class BaseLogicAPI(APIBase):
    def __init__(self):
        APIBase.__init__(self)

    def minimum(self, a, b):
        raise NotImplementedError

    def maximum(self, a, b):
        raise NotImplementedError

    def where(self, cond, true, false):
        if self.dtype_of(cond) != 'bool':
            cond = self.less(0, self.abs(cond))
        return cond * true + (1 - cond) * false

    def equal(self, a, b, dtype=None):
        raise NotImplementedError

    def not_equal(self, a, b, dtype=None):
        raise NotImplementedError

    def less(self, a, b, dtype=None):
        raise NotImplementedError

    def less_equal(self, a, b, dtype=None):
        raise NotImplementedError

    def greater_equal(self, a, b, dtype=None):
        raise NotImplementedError

    def greater(self, a, b, dtype=None):
        raise NotImplementedError
