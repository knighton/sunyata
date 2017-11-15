import mxnet as mx

from .....base.layer.shape.pad.reflect import BaseReflectPadAPI


class MXNetReflectPadAPI(BaseReflectPadAPI):
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

    def reflect_pad1d(self, x, pad):
        x = mx.nd.expand_dims(x, 2)
        (left, right), = self.unpack_pad(pad, 1)
        pad = (0, 0), (left, right)
        x = self.reflect_pad2d(x, pad)
        return self.squeeze(x, 2)

    def reflect_pad2d(self, x, pad):
        (top, bottom), (left, right) = self.unpack_pad(pad, 2)
        mx_pad = 0, 0, 0, 0, top, bottom, left, right
        return mx.nd.pad(x, 'reflect', mx_pad)

    def reflect_pad3d(self, x, pad):
        (front, back), (top, bottom), (left, right) = self.unpack_pad(pad, 3)
        mx_pad = 0, 0, 0, 0, front, back, top, bottom, left, right
        return mx.nd.pad(x, 'reflect', mx_pad)
