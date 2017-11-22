from .... import backend as Z
from .base import PadLayer, PadSpec


class EdgePadLayer(PadLayer):
    def __init__(self, pad, ndim):
        super().__init__(pad, ndim)

    def forward_one(self, x, is_training):
        assert self.ndim(x) == self.ndim
        return Z.edge_pad(x, self.pad)


class EdgePadSpec(PadSpec):
    def __init__(self, pad, ndim=None):
        super().__init__(pad, ndim)

    def make_layer(self, ndim):
        return EdgePadLayer(self.pad, self.ndim)
