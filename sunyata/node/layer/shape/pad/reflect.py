from ..... import backend as Z
from .base import PadLayer, PadSpec


class ReflectPadLayer(PadLayer):
    def __init__(self, pad, x_ndim=None):
        super().__init__(pad, x_ndim)

    def transform(self, x, train):
        return Z.reflect_pad(x, self.pad)


class ReflectPadSpec(PadSpec):
    def __init__(self, pad, spatial_ndim=None):
        super().__init__(pad, spatial_ndim)

    def make_layer(self, form):
        return ReflectPadLayer(self.pad, self.x_ndim())
