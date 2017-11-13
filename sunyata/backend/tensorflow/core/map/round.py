import tensorflow as tf

from ....base.core.map.round import BaseRoundAPI


class TensorFlowRoundAPI(BaseRoundAPI):
    def __init__(self):
        BaseRoundAPI.__init__(self)

    def ceil(self, x):
        return tf.ceil(x)

    def floor(self, x):
        return tf.floor(x)

    def round(self, x):
        return tf.round(x)
