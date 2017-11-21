import numpy as np
import tensorflow as tf

from .....base.layer.shape.upsample.linear import BaseLinearUpsampleAPI


class TensorFlowLinearUpsampleAPI(BaseLinearUpsampleAPI):
    def __init__(self):
        BaseLinearUpsampleAPI.__init__(self)
        self._ndim2linear_upsample = {
            1: self.linear_upsample1d,
            2: self.linear_upsample2d,
            3: self.linear_upsample3d,
        }

    def linear_upsample(self, x, scale):
        ndim = self.ndim(x) - 2
        return self._ndim2linear_upsample[ndim](x, scale)

    def linear_upsample1d(self, x, scale):
        scale = self.to_shape(scale, 1)
        scale = (1,) + scale
        x = tf.squeeze(x, 2)
        x = self.linear_upsample2d(x, scale)
        return tf.unsqueeze(x, 2)

    def linear_upsample2d(self, x, scale):
        scale = self.to_shape(scale, 2)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(np.array(scale)).astype('int32')
        x = tf.transpose(x, (0, 2, 3, 1))
        x = tf.image.resize_bilinear(x, new_shape)
        return tf.transpose(x, (0, 3, 1, 2))

    def linear_upsample3d(self, x, scale):
        raise NotImplementedError
