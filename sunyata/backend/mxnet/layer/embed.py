import mxnet as mx

from ...base.layer.embed import BaseEmbedAPI


class MXNetEmbedAPI(BaseEmbedAPI):
    def __init__(self):
        BaseEmbedAPI.__init__(self)

    def embed(self, x, reference):
        vocab_size, channels = reference.shape
        channels_last = mx.nd.Embedding(
            data=x, weight=reference, input_dim=vocab_size, output_dim=channels,
            dtype=reference.dtype)
        ndim = channels_last.ndim
        axes = (0, ndim - 1) + tuple(range(1, ndim - 1))
        return mx.nd.transpose(channels_last, axes)
