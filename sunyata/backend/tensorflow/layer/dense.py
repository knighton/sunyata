import tensorflow as tf

from ...base.layer.dense import BaseDenseAPI


class TensorFlowDenseAPI(BaseDenseAPI):
    def dense(self, x, kernel, bias):
        return tf.matmul(x, kernel) + bias
