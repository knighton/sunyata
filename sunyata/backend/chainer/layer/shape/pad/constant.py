from chainer import functions as F
from chainer import Variable
import numpy as np

from .....base.layer.shape.pad.constant import BaseConstantPadAPI


class ChainerConstantPadAPI(BaseConstantPadAPI):
    def __init__(self):
        BaseConstantPadAPI.__init__(self)
        self._ndim2constant_pad = {
            1: self.constant_pad1d,
            2: self.constant_pad2d,
            3: self.constant_pad3d,
        }

    def constant_pad(self, x, pad, value):
        ndim = x.ndim - 2
        return self._ndim2constant_pad[ndim](x, pad, value)

    def _constant_pad(self, x, pad, value):
        pad = ((0, 0), (0, 0)) + self.unpack_int_pad(pad, x.ndim)
        if isinstance(x, Variable):
            func = F.pad
        else:
            func = np.pad
        return func(x, pad, 'constant', constant_values=value)

    def constant_pad1d(self, x, pad, value):
        return self._constant_pad(x, pad, value)

    def constant_pad2d(self, x, pad, value):
        return self._constant_pad(x, pad, value)

    def constant_pad3d(self, x, pad, value):
        return self._constant_pad(x, pad, value)
