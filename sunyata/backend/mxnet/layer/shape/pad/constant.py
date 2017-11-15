import mxnet as mx

from .....base.layer.shape.pad.constant import BaseConstantPadAPI


class MXNetConstantPadAPI(BaseConstantPadAPI):
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

    def constant_pad1d(self, x, pad, value):
        x = mx.nd.expand_dims(x, 2)
        (left, right), = self.unpack_int_pad(pad, 1)
        pad = (0, 0), (left, right)
        x = self.constant_pad2d(x, pad, value)
        return self.squeeze(x, 2)

    def constant_pad2d(self, x, pad, value):
        (top, bottom), (left, right) = self.unpack_int_pad(pad, 2)
        mx_pad = 0, 0, 0, 0, top, bottom, left, right
        return mx.nd.pad(x, 'constant', mx_pad, value)

    def constant_pad3d(self, x, pad, value):
        (front, back), (top, bottom), (left, right) = \
            self.unpack_int_pad(pad, 3)
        mx_pad = 0, 0, 0, 0, front, back, top, bottom, left, right
        return mx.nd.pad(x, 'constant', mx_pad, value)
