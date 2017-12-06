from ... import backend as Z
from ..base import TransformLayer, TransformSpec


class GaussianNoiseLayer(TransformLayer):
    def __init__(self, std, keep_axis, ndim):
        super().__init__(ndim)
        self.std = std
        self.keep_axis = keep_axis

    def forward_one(self, x, is_training):
        return Z.gaussian_noise(x, is_training, self.std, self.keep_axis)


class GaussianNoiseSpec(TransformSpec):
    def __init__(self, std, keep_axis=None, ndim=None):
        super().__init__(ndim)
        self.std = std
        self.keep_axis = keep_axis

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        layer = GaussianNoiseLayer(self.std, self.keep_axis, ndim)
        return layer, form
