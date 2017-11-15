from chainer import functions as F
from chainer import Variable
import numpy as np

from .....base.layer.shape.pad.reflect import BaseReflectPadAPI


class ChainerReflectPadAPI(BaseReflectPadAPI):
    def __init__(self):
        BaseReflectPadAPI.__init__(self)
        self._ndim2reflect_pad = {
            1: self.reflect_pad1d,
            2: self.reflect_pad2d,
            3: self.reflect_pad3d,
        }

    def reflect_pad(self, x, pad):
        ndim = x.ndim - 2
        return self._ndim2reflect_pad[ndim](x, pad)

    def _reflect_pad(self, x, pad):
        pad = ((0, 0), (0, 0)) + self.unpack_int_pad(pad, x.ndim)
        if isinstance(x, Variable):
            func = F.pad
        else:
            func = np.pad
        return func(x, pad, 'reflect')

    def reflect_pad1d(self, x, pad):
        return self._reflect_pad(x, pad)

    def reflect_pad2d(self, x, pad):
        return self._reflect_pad(x, pad)

    def reflect_pad3d(self, x, pad):
        return self._reflect_pad(x, pad)
