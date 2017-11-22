import tensorflow as tf

from ...base.layer.activation import BaseActivationAPI


class TensorFlowActivationAPI(BaseActivationAPI):
    def log_softmax(self, x):
        return tf.nn.log_softmax(x)

    def selu(self, x):
        return tf.nn.selu(x)

    def sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def softmax(self, x):
        return tf.nn.softmax(x)

    def softsign(self, x):
        return tf.nn.softsign(x)
