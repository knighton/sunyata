import tensorflow as tf

from ....base.core.map.power import BasePowerAPI


class TensorFlowPowerAPI(BasePowerAPI):
    def __init__(self):
        BasePowerAPI.__init__(self)

    def pow(self, x, a):
        return tf.power(x, a)

    def rsqrt(self, x):
        return tf.rsqrt(x)

    def sqrt(self, x):
        return tf.sqrt(x)

    def square(self, x):
        return tf.square(x)
