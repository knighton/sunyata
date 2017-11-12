from ...base.core.logic import BaseLogicAPI

import tensorflow as tf


class TensorFlowLogicAPI(BaseLogicAPI):
    def __init__(self):
        BaseLogicAPI.__init__(self)

    def minimum(self, a, b):
        return tf.minimum(a, b)

    def maximum(self, a, b):
        return tf.maximum(a, b)

    def equal(self, a, b, dtype=None):
        x = tf.equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def not_equal(self, a, b, dtype=None):
        x = tf.not_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def less(self, a, b, dtype=None):
        x = tf.less(a, b)
        return self._cast_bool_output(a, x, dtype)

    def less_equal(self, a, b, dtype=None):
        x = tf.less_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def greater_equal(self, a, b, dtype=None):
        x = tf.greater_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def greater(self, a, b, dtype=None):
        x = tf.greater(a, b)
        return self._cast_bool_output(a, x, dtype)
