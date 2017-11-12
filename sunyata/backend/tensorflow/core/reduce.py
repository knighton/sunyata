import tensorflow as tf

from ...base.core.reduce import BaseReduceAPI


class TensorFlowReduceAPI(BaseReduceAPI):
    def __init__(self):
        BaseReduceAPI.__init__(self)

    def argmin(self, x, axis=-1):
        return tf.argmin(x, axis)

    def argmax(self, x, axis=-1):
        return tf.argmax(x, axis)

    def min(self, x, axis=None, keepdims=False):
        return tf.reduce_min(x, axis, keepdims)

    def max(self, x, axis=None, keepdims=False):
        return tf.reduce_max(x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return tf.reduce_sum(x, axis, keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return tf.reduce_mean(x, axis, keepdims)

    def prod(self, x, axis=None, keepdims=False):
        return tf.reduce_prod(x, axis, keepdims)

    def any(self, x, axis=None, keepdims=False):
        return tf.reduce_any(x, axis, keepdims)

    def all(self, x, axis=None, keepdims=False):
        return tf.reduce_all(x, axis, keepdims)
