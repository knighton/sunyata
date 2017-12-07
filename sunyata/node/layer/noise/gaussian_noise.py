from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class GaussianNoiseLayer(TransformLayer):
    def __init__(self, std, keep_axis, x_ndim=None):
        super().__init__(x_ndim)
        self.std = std
        self.keep_axis = keep_axis

    def transform(self, x, train):
        return Z.gaussian_noise(x, train, self.std, self.keep_axis)


class GaussianNoiseSpec(TransformSpec):
    def __init__(self, std, keep_axis=None, spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.std = std
        self.keep_axis = keep_axis

    def build_transform(self, form):
        layer = GaussianNoiseLayer(self.std, self.keep_axis, self.x_ndim())
        return layer, form


node_wrap(GaussianNoiseSpec, (None, 0, 1, 2, 3))
