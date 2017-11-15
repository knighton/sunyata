import numpy as np
import torch
from torch.autograd import Variable
from torch import _TensorBase

from ...base.core.logic import BaseLogicAPI


class PyTorchLogicAPI(BaseLogicAPI):
    def __init__(self):
        BaseLogicAPI.__init__(self)

    def _min_max_analyze(self, x):
        if isinstance(x, Variable):
            kind = 5
            dtype = self.dtype_of(x)
            ndim = self.ndim(x)
            device = self.device_of(x)
        elif isinstance(x, _TensorBase):
            kind = 4
            dtype = self.dtype_of(x)
            ndim = self.ndim(x)
            device = self.device_of(x)
        elif isinstance(x, np.number):
            kind = 3
            dtype = x.dtype.name
            ndim = 0
            device = None
        elif isinstance(x, float):
            kind = 2
            dtype = None
            ndim = 0
            device = None
        elif isinstance(x, int):
            kind = 1
            dtype = None
            ndim = 0
            device = None
        else:
            assert False
        return kind, dtype, ndim, device

    def _min_max_choose_kind(self, a_kind, b_kind):
        return max(a_kind, b_kind, 4)

    def _min_max_choose_dtype(self, a_kind, a_dtype, b_kind, b_dtype):
        low, high = sorted([a_kind, b_kind])
        if low == 1:
            if high == 1:
                dtype = self.intx()
            elif high == 2:
                dtype = self.floatx()
            else:
                dtype = a_dtype if b_dtype is None else b_dtype
        elif low == 2:
            if high == 2:
                dtype = self.floatx()
            else:
                dtype = a_dtype if b_dtype is None else b_dtype
        else:
            assert a_dtype == b_dtype
            dtype = a_dtype
        return dtype

    def _min_max_choose_ndim(self, a_ndim, b_ndim):
        if a_ndim:
            if b_ndim:
                assert a_ndim == b_ndim
                ndim = a_ndim
            else:
                ndim = a_ndim
        else:
            if b_ndim:
                ndim = b_ndim
            else:
                ndim = 0
        return ndim

    def _min_max_choose_device(self, a_device, b_device):
        if a_device is None:
            if b_device is None:
                device = self.current_device()
            else:
                device = b_device
        else:
            if b_device is None:
                device = a_device
            else:
                assert a_device is b_device
                device = a_device
        return device

    def _min_max_choose(self, a_kind, a_dtype, a_ndim, a_device, b_kind,
                        b_dtype, b_ndim, b_device):
        to_kind = self._min_max_choose_kind(a_kind, b_kind)
        to_dtype = self._min_max_choose_dtype(a_kind, a_dtype, b_kind, b_dtype)
        to_ndim = self._min_max_choose_ndim(a_ndim, b_ndim)
        to_device = self._min_max_choose_device(a_device, b_device)
        return to_kind, to_dtype, to_ndim, to_device

    def _min_max_load(self, x, to_kind, to_dtype, to_ndim, to_device):
        if isinstance(x, (np.number, float, int)):
            to_shape = (1,) * to_ndim
            x = np.full(to_shape, x, to_dtype)
            x = self.cast_numpy_to_device(x, to_dtype, to_device)
            if to_kind == 5:
                x = Variable(x)
        elif isinstance(x, _TensorBase) and to_kind == 5:
            x = Variable(x)
        return x

    def _min_max(self, func_name, a, b):
        a_kind, a_dtype, a_ndim, a_device = self._min_max_analyze(a)
        b_kind, b_dtype, b_ndim, b_device = self._min_max_analyze(b)
        to_kind, to_dtype, to_ndim, to_device = self._min_max_choose(
            a_kind, a_dtype, a_ndim, a_device, b_kind, b_dtype, b_ndim,
            b_device)
        a = self._min_max_load(a, to_kind, to_dtype, to_ndim, to_device)
        b = self._min_max_load(b, to_kind, to_dtype, to_ndim, to_device)
        return getattr(torch, func_name)(a, b)

    def minimum(self, a, b):
        return self._min_max('min', a, b)

    def maximum(self, a, b):
        return self._min_max('max', a, b)

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
