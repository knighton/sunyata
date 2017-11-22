from .... import backend as Z
from .base import PadLayer, PadSpec


class ReflectPadLayer(PadLayer):
    def __init__(self, pad, ndim):
        super().__init__(pad, ndim)

    def forward_one(self, x, is_training):
        assert self.ndim(x) == self.ndim
        return Z.reflect_pad(x, self.pad)


class ReflectPadSpec(PadSpec):
    def __init__(self, pad, ndim=None):
        super().__init__(pad, ndim)

    def make_layer(self, ndim):
        return ReflectPadLayer(self.pad, self.ndim)
