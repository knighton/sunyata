import tensorflow as tf

from .....base.layer.shape.pad.reflect import BaseReflectPadAPI


class TensorFlowReflectPadAPI(BaseReflectPadAPI):
    def __init__(self):
        BaseReflectPadAPI.__init__(self)
        self._ndim2reflect_pad = {
            1: self.reflect_pad1d,
            2: self.reflect_pad2d,
            3: self.reflect_pad3d,
        }

    def reflect_pad(self, x, pad, value):
        ndim = len(x.shape) - 2
        return self._ndim2reflect_pad[ndim](x, pad, value)

    def reflect_pad1d(self, x, pad, value):
        (left, right), = self.unpack_int_pad(pad, 1)
        pad = (0, 0), (0, 0), (left, right)
        return tf.pad(x, pad, 'reflect')

    def reflect_pad2d(self, x, pad, value):
        (top, bottom), (left, right) = self.unpack_int_pad(pad, 2)
        pad = (0, 0), (0, 0), (top, bottom), (left, right)
        return tf.pad(x, pad, 'reflect')

    def reflect_pad3d(self, x, pad, value):
        (front, back), (top, bottom), (left, right) = \
            self.unpack_int_pad(pad, 2)
        pad = (0, 0), (0, 0), (front, back), (top, bottom), (left, right)
        return tf.pad(x, pad, 'reflect')
