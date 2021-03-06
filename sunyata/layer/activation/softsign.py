from ... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftSignLayer(TransformLayer):
    def transform(self, x, train):
        return Z.softsign(x)


class SoftSignSpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        return SoftSignLayer(self.x_ndim()), form


node_wrap(SoftSignSpec)
