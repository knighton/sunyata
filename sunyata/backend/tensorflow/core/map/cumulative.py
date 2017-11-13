import tensorflow as tf

from ....base.core.map.cumulative import BaseCumulativeAPI


class TensorFlowCumulativeAPI(BaseCumulativeAPI):
    def __init__(self):
        BaseCumulativeAPI.__init__(self)

    def cumsum(self, x, axis):
        return tf.cumsum(x, axis)

    def cumprod(self, x, axis):
        return tf.cumprod(x, axis)
