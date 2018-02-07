from .... import backend as Z
from ...base import node_wrap
from .base import PadLayer, PadSpec


class ConstantPadLayer(PadLayer):
    def __init__(self, pad, value, x_ndim=None):
        super().__init__(pad, x_ndim)
        self.value = value

    def transform(self, x, train):
        return Z.constant_pad(x, self.pad, self.value)


class ConstantPadSpec(PadSpec):
    def __init__(self, pad, value, spatial_ndim=None):
        super().__init__(pad, spatial_ndim)
        self.value = value

    def make_layer(self, form):
        return ConstantPadLayer(self.pad, self.value, self.x_ndim())


node_wrap(ConstantPadSpec)
