import numpy as np
import tensorflow as tf

from .....base.layer.shape.upsample.nearest import BaseNearestUpsampleAPI


class TensorFlowNearestUpsampleAPI(BaseNearestUpsampleAPI):
    def __init__(self):
        BaseNearestUpsampleAPI.__init__(self)
        self._ndim2nearest_upsample = {
            1: self.nearest_upsample1d,
            2: self.nearest_upsample2d,
            3: self.nearest_upsample3d,
        }

    def nearest_upsample(self, x, scale):
        ndim = self.ndim(x) - 2
        return self._ndim2nearest_upsample[ndim](x, scale)

    def nearest_upsample1d(self, x, scale):
        x = tf.squeeze(x, 2)
        x = self.nearest_upsample2d(x, scale)
        return tf.unsqueeze(x, 2)

    def nearest_upsample2d(self, x, scale):
        scale = self.to_shape(scale, 2)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(np.array(scale)).astype('int32')
        x = tf.transpose(x, (0, 2, 3, 1))
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        return tf.transpose(x, (0, 3, 1, 2))

    def nearest_upsample3d(self, x, scale):
        assert self.ndim(x) == 5
        scale = self.to_shape(scale, 3)
        return self.repeat(x, (1, 1) + scale)
