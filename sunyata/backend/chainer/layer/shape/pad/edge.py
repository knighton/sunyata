from chainer import functions as F
from chainer import Variable
import numpy as np

from .....base.layer.shape.pad.edge import BaseEdgePadAPI


class ChainerEdgePadAPI(BaseEdgePadAPI):
    def __init__(self):
        BaseEdgePadAPI.__init__(self)
        self._ndim2edge_pad = {
            1: self.edge_pad1d,
            2: self.edge_pad2d,
            3: self.edge_pad3d,
        }

    def edge_pad(self, x, pad):
        ndim = x.ndim - 2
        return self._ndim2edge_pad[ndim](x, pad)

    def _edge_pad(self, x, pad):
        pad = ((0, 0), (0, 0)) + self.unpack_int_pad(pad, x.ndim)
        if isinstance(x, Variable):
            func = F.pad
        else:
            func = np.pad
        return func(x, pad, 'edge')

    def edge_pad1d(self, x, pad):
        return self._edge_pad(x, pad)

    def edge_pad2d(self, x, pad):
        return self._edge_pad(x, pad)

    def edge_pad3d(self, x, pad):
        return self._edge_pad(x, pad)
