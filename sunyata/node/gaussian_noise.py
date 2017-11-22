from .. import backend as Z
from .base import TransformLayer, TransformSpec


class GaussianNoiseLayer(TransformLayer):
    def __init__(self, std, keep_axis, ndim):
        self.std = std
        self.keep_axis = keep_axis
        self.ndim = ndim

    def forward_one(self, x, is_training):
        assert Z.ndim(x) == self.ndim
        return Z.gaussian_noise(x, is_training, self.std, self.keep_axis)


class GaussianNoiseSpec(TransformSpec):
    def __init__(self, std, keep_axis=None, ndim=None):
        self.std = std
        self.keep_axis = keep_axis
        self.ndim = ndim

    def build_one(self, form):
        if self.ndim is None:
            ndim = len(form.shape) + 1
        else:
            ndim = self.ndim
            assert len(form.shape) + 1 == ndim
        layer = GaussianNoiseLayer(self.std, self.keep_axis, ndim)
        return layer, form
