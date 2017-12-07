from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftPlusLayer(TransformLayer):
    def transform(self, x, train):
        return Z.softplus(x)


class SoftPlusSpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        return SoftPlusLayer(self.x_ndim()), form


node_wrap(SoftPlusSpec)
