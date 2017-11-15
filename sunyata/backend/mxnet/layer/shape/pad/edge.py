import mxnet as mx

from .....base.layer.shape.pad.edge import BaseEdgePadAPI


class MXNetEdgePadAPI(BaseEdgePadAPI):
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

    def edge_pad1d(self, x, pad):
        x = mx.nd.expand_dims(x, 2)
        (left, right), = self.unpack_int_pad(pad, 1)
        pad = (0, 0), (left, right)
        x = self.edge_pad2d(x, pad)
        return self.squeeze(x, 2)

    def edge_pad2d(self, x, pad):
        (top, bottom), (left, right) = self.unpack_int_pad(pad, 2)
        mx_pad = 0, 0, 0, 0, top, bottom, left, right
        return mx.nd.pad(x, 'edge', mx_pad)

    def edge_pad3d(self, x, pad):
        (front, back), (top, bottom), (left, right) = \
            self.unpack_int_pad(pad, 3)
        mx_pad = 0, 0, 0, 0, front, back, top, bottom, left, right
        return mx.nd.pad(x, 'edge', mx_pad)
