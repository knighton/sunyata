from .... import backend as Z
from ...base import node_wrap
from .base import PadLayer, PadSpec


class EdgePadLayer(PadLayer):
    def __init__(self, pad, x_ndim=None):
        super().__init__(pad, x_ndim)

    def transform(self, x, train):
        return Z.edge_pad(x, self.pad)


class EdgePadSpec(PadSpec):
    def __init__(self, pad, spatial_ndim=None):
        super().__init__(pad, spatial_ndim)

    def make_layer(self, form):
        return EdgePadLayer(self.pad, self.x_ndim())


node_wrap(EdgePadSpec)
