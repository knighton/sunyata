from .... import backend as Z
from .base import PadLayer, PadSpec


class ConstantPadLayer(PadLayer):
    def __init__(self, pad, value, ndim):
        super().__init__(pad, ndim)
        self.value = value

    def forward_one(self, x, is_training):
        return Z.constant_pad(x, self.pad, self.value)


class ConstantPadSpec(PadSpec):
    def __init__(self, pad, value, ndim=None):
        super().__init__(pad, ndim)
        self.value = value

    def make_layer(self, form):
        ndim = self.in_ndim(form.shape)
        return ConstantPadLayer(self.pad, self.value, ndim)
