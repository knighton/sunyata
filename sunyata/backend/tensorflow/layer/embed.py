import tensorflow as tf

from ...base.layer.embed import BaseEmbedAPI


class TensorFlowEmbedAPI(BaseEmbedAPI):
    def __init__(self):
        BaseEmbedAPI.__init__(self)

    def embed(self, x, reference):
        channels_last = tf.gather(reference, x)
        ndim = len(channels_last.shape)
        axes = (0, ndim - 1) + tuple(range(1, ndim - 1))
        return tf.transpose(channels_last, axes)
