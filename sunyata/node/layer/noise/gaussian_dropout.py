from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class GaussianDropoutLayer(TransformLayer):
    def __init__(self, rate, keep_axis, x_ndim=None):
        super().__init__(x_ndim)
        self.rate = rate
        self.keep_axis = keep_axis

    def transform(self, x, is_training):
        return Z.gaussian_dropout(x, is_training, self.rate, self.keep_axis)


class GaussianDropoutSpec(TransformSpec):
    def __init__(self, rate=0.5, keep_axis=None, spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.rate = rate
        self.keep_axis = keep_axis

    def build_transform(self, form):
        layer = GaussianDropoutLayer(self.rate, self.keep_axis, self.x_ndim())
        return layer, form


node_wrap(GaussianDropoutSpec, (None, 0, 1, 2, 3))
