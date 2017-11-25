import tensorflow as tf

from ...base.core.linalg import BaseLinearAlgebraAPI


class TensorFlowLinearAlgebraAPI(BaseLinearAlgebraAPI):
    def matmul(self, a, b):
        return tf.matmul(a, b)
