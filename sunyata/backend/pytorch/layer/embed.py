from ...base.layer.embed import BaseEmbedAPI


class PyTorchEmbedAPI(BaseEmbedAPI):
    def __init__(self):
        BaseEmbedAPI.__init__(self)

    def embed(self, x, reference):
        channels_last = reference.index_select(0, x.view(-1))
        channels_last = channels_last.view(x.size() + (-1,))
        ndim = channels_last.dim()
        axes = (0, ndim - 1) + tuple(range(1, ndim - 1))
        return channels_last.permute(*axes).contiguous()
