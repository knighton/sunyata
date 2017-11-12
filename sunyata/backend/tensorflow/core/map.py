import numpy as np
import tensorflow as tf

from ...base.core.map import BaseMapAPI


class TensorFlowMapAPI(BaseMapAPI):
    def __init__(self):
        BaseMapAPI.__init__(self)

    def abs(self, x):
        return tf.abs(x)

    def neg(self, x):
        return tf.negative(x)

    def sign(self, x):
        return tf.sign(x)

    def clip(self, x, min=-np.inf, max=np.inf):
        return tf.clip_by_value(x, min, max)

    def log(self, x):
        return tf.log(x)

    def pow(self, x, a):
        return tf.pow(x, a)
