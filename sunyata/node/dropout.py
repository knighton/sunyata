from .. import backend as Z
from .base import TransformLayer, TransformSpec


class VanillaDropoutLayer(TransformLayer):
    def __init__(self, rate, ndim):
        self.rate = rate
        self.ndim = ndim

    def forward_one(self, x, is_training):
        assert Z.ndim(x) == self.ndim
        return Z.vanilla_dropout(x, is_training, self.rate)


class VanillaDropoutSpec(TransformSpec):
    def __init__(self, rate=0.5, ndim=None):
        self.rate = rate
        self.ndim = ndim

    def build_one(self, form):
        if self.ndim is None:
            ndim = len(form.shape) + 1
        else:
            ndim = self.ndim
            assert len(form.shape) + 1 == ndim
        layer = VanillaDropoutLayer(self.rate, ndim)
        return layer, form


class SpatialDropoutLayer(TransformLayer):
    def __init__(self, rate, keep_axis, ndim):
        self.rate = rate
        self.keep_axis = keep_axis
        self.ndim = ndim

    def forward_one(self, x, is_training):
        assert Z.ndim(x) == self.ndim
        return Z.spatial_dropout(x, is_training, self.keep_axis, self.rate)


class SpatialDropoutSpec(TransformSpec):
    def __init__(self, rate=0.5, keep_axis=1, ndim=None):
        self.rate = rate
        self.keep_axis = keep_axis
        self.ndim = ndim

    def build_one(self, form):
        if self.ndim is None:
            ndim = len(form.shape) + 1
        else:
            ndim = self.ndim
            assert len(form.shape) + 1 == ndim
        return SpatialDropoutLayer(self.rate, self.keep_axis, ndim)
