import tensorflow as tf

from ....base.core.map.sign import BaseSignAPI


class TensorFlowSignAPI(BaseSignAPI):
    def __init__(self):
        BaseSignAPI.__init__(self)

    def abs(self, x):
        return tf.abs(x)

    def neg(self, x):
        return tf.negative(x)

    def sign(self, x):
        return tf.sign(x)
