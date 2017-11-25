from chainer import functions as F

from ...base.layer.embed import BaseEmbedAPI


class ChainerEmbedAPI(BaseEmbedAPI):
    def __init__(self):
        BaseEmbedAPI.__init__(self)

    def embed(self, x, reference):
        channels_last = F.embed_id(x, reference)
        ndim = channels_last.ndim
        axes = (0, ndim - 1) + tuple(range(1, ndim - 1))
        return F.transpose(channels_last, axes)
