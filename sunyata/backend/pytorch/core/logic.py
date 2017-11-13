import torch

from ...base.core.logic import BaseLogicAPI


class PyTorchLogicAPI(BaseLogicAPI):
    def __init__(self):
        BaseLogicAPI.__init__(self)

    def minimum(self, a, b):
        return torch.min(a, b)

    def maximum(self, a, b):
        return torch.max(a, b)

    def _compare(self, a, x):
        return self.cast(x, self.dtype_of(a))

    def equal(self, a, b):
        return self._compare(a, a == b)

    def not_equal(self, a, b):
        return self._compare(a, a != b)

    def less(self, a, b):
        return self._compare(a, a < b)

    def less_equal(self, a, b):
        return self._compare(a, a <= b)

    def greater_equal(self, a, b):
        return self._compare(a, a >= b)

    def greater(self, a, b):
        return self._compare(a, a > b)
