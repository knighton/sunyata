from ...base.core.logic import BaseLogicAPI

import tensorflow as tf


class TensorFlowLogicAPI(BaseLogicAPI):
    def __init__(self):
        BaseLogicAPI.__init__(self)

    def minimum(self, a, b):
        return tf.minimum(a, b)

    def maximum(self, a, b):
        return tf.maximum(a, b)

    def _compare(self, func, a, b):
        return tf.cast(func(a, b), a.dtype)

    def equal(self, a, b):
        return self._compare(tf.equal, a, b)

    def not_equal(self, a, b):
        return self._compare(tf.not_equal, a, b)

    def less(self, a, b):
        return self._compare(tf.less, a, b)

    def less_equal(self, a, b):
        return self._compare(tf.less_equal, a, b)

    def greater_equal(self, a, b):
        return self._compare(tf.greater_equal, a, b)

    def greater(self, a, b):
        return self._compare(tf.greater, a, b)
