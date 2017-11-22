from .... import backend as Z
from .base import PadLayer, PadSpec


class EdgePadLayer(PadLayer):
    def __init__(self, pad, ndim):
        super().__init__(pad, ndim)

    def forward_one(self, x, is_training):
        return Z.edge_pad(x, self.pad)


class EdgePadSpec(PadSpec):
    def __init__(self, pad, ndim=None):
        super().__init__(pad, ndim)

    def make_layer(self, form):
        ndim = self.in_ndim(form.shape)
        return EdgePadLayer(self.pad, ndim)
