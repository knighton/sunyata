from ... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class DropoutLayer(TransformLayer):
    def __init__(self, rate, keep_axis, x_ndim=None):
        super().__init__(x_ndim)
        self.rate = rate
        self.keep_axis = keep_axis

    def transform(self, x, train):
        return Z.dropout(x, train, self.rate, self.keep_axis)


class DropoutSpec(TransformSpec):
    def __init__(self, rate=0.5, keep_axis=None, spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.rate = rate
        self.keep_axis = keep_axis

    def build_transform(self, form):
        layer = DropoutLayer(self.rate, self.keep_axis, self.x_ndim())
        return layer, form


node_wrap(DropoutSpec, (None, 0, 1, 2, 3))
