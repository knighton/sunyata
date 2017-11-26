import tensorflow as tf

from ....base.layer.dot.dense import BaseDenseAPI


class TensorFlowDenseAPI(BaseDenseAPI):
    def __init__(self):
        BaseDenseAPI.__init__(self)

    def dense(self, x, kernel, bias):
        x = tf.matmul(kernel, x)
        if bias is not None:
            x += bias
        return x
