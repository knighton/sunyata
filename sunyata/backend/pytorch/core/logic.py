import torch

from ...base.core.logic import BaseLogicAPI


class PyTorchLogicAPI(BaseLogicAPI):
    def __init__(self):
        BaseLogicAPI.__init__(self)

    def minimum(self, a, b):
        return torch.min(a, b)

    def maximum(self, a, b):
        return torch.max(a, b)

    def equal(self, a, b, dtype=None):
        return self._cast_bool_output(a, a == b, dtype)

    def not_equal(self, a, b, dtype=None):
        return self._cast_bool_output(a, a != b, dtype)

    def less(self, a, b, dtype=None):
        return self._cast_bool_output(a, a < b, dtype)

    def less_equal(self, a, b, dtype=None):
        return self._cast_bool_output(a, a <= b, dtype)

    def greater_equal(self, a, b, dtype=None):
        return self._cast_bool_output(a, a >= b, dtype)

    def greater(self, a, b, dtype=None):
        return self._cast_bool_output(a, a > b, dtype)
