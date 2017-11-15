import tensorflow as tf

from .....base.layer.shape.pad.constant import BaseConstantPadAPI


class TensorFlowConstantPadAPI(BaseConstantPadAPI):
    def __init__(self):
        BaseConstantPadAPI.__init__(self)
        self._ndim2constant_pad = {
            1: self.constant_pad1d,
            2: self.constant_pad2d,
            3: self.constant_pad3d,
        }

    def constant_pad(self, x, pad, value):
        ndim = len(x.shape) - 2
        return self._ndim2constant_pad[ndim](x, pad, value)

    def constant_pad1d(self, x, pad, value):
        (left, right), = self.unpack_int_pad(pad, 1)
        pad = (0, 0), (0, 0), (left, right)
        return tf.pad(x, pad, 'constant', constant_values=value)

    def constant_pad2d(self, x, pad, value):
        (top, bottom), (left, right) = self.unpack_int_pad(pad, 2)
        pad = (0, 0), (0, 0), (top, bottom), (left, right)
        return tf.pad(x, pad, 'constant', constant_values=value)

    def constant_pad3d(self, x, pad, value):
        (front, back), (top, bottom), (left, right) = \
            self.unpack_int_pad(pad, 2)
        pad = (0, 0), (0, 0), (front, back), (top, bottom), (left, right)
        return tf.pad(x, pad, 'constant', constant_values=value)
