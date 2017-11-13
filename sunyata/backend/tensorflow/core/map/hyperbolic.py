import tensorflow as tf

from ....base.core.map.hyperbolic import BaseHyperbolicAPI


class TensorFlowHyperbolicAPI(BaseHyperbolicAPI):
    def __init__(self):
        BaseHyperbolicAPI.__init__(self)

    def sinh(self, x):
        return tf.sinh(x)

    def cosh(self, x):
        return tf.cosh(x)

    def tanh(self, x):
        return tf.tanh(x)

    def arcsinh(self, x):
        return tf.asinh(x)

    def arccosh(self, x):
        return tf.acosh(x)

    def arctanh(self, x):
        return tf.atanh(x)
