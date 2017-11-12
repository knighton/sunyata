import tensorflow as tf

from ...base.layer.activation import BaseActivationAPI


class TensorFlowActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return tf.nn.softmax(x)
