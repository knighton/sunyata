import tensorflow as tf

from ...base.core.shape import BaseShapeAPI


class TensorFlowShapeAPI(BaseShapeAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)

    def ndim(self, x):
        return len(x.shape)

    def shape(self, x):
        return tuple(map(int, x.shape))

    def size(self, x):
        return int(tf.size(x).numpy())

    def reshape(self, x, shape):
        return tf.reshape(x, shape)

    def expand_dims(self, x, axis):
        return tf.expand_dims(x, axis)
