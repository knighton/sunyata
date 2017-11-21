import tensorflow as tf

from ...base.core.shape import BaseShapeAPI


class TensorFlowShapeAPI(BaseShapeAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)

    def ndim(self, x):
        return len(x.shape)

    def shape(self, x):
        return tuple(x.get_shape().as_list())

    def size(self, x):
        return x.get_shape().num_elements()

    def reshape(self, x, shape):
        return tf.reshape(x, shape)

    def expand_dims(self, x, axis):
        return tf.expand_dims(x, axis)

    def squeeze(self, x, axis=None):
        return tf.squeeze(x, axis)

    def tile_axis(self, x, axis, repeat):
        repeats = [1] * self.ndim(x)
        repeats[axis] = repeat
        return tf.tile(x, repeats)

    def tile(self, x, repeats):
        return tf.tile(x, repeats)

    def transpose(self, x, axes):
        return tf.transpose(x, axes)

    def split(self, x, axis):
        return tf.split(x, x.shape[axis], axis)

    def concat(self, xx, axis):
        return tf.concat(xx, axis)

    def stack(self, xx, axis=0):
        return tf.stack(xx, axis)
