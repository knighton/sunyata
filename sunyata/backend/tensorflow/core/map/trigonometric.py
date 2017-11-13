import tensorflow as tf

from ....base.core.map.trigonometric import BaseTrigonometricAPI


class TensorFlowTrigonometricAPI(BaseTrigonometricAPI):
    def __init__(self):
        BaseTrigonometricAPI.__init__(self)

    def sin(self, x):
        return tf.sin(x)

    def cos(self, x):
        return tf.cos(x)

    def tan(self, x):
        return tf.tan(x)

    def arcsin(self, x):
        return tf.asin(x)

    def arccos(self, x):
        return tf.acos(x)

    def arctan(self, x):
        return tf.atan(x)
