import tensorflow as tf

from ....base.core.map.log_exp import BaseLogExpAPI


class TensorFlowLogExpAPI(BaseLogExpAPI):
    def __init__(self):
        BaseLogExpAPI.__init__(self)

    def exp(self, x):
        return tf.exp(x)

    def expm1(self, x):
        return tf.expm1(x)

    def log(self, x):
        return tf.log(x)

    def log1p(self, x):
        return tf.log1p(x)
