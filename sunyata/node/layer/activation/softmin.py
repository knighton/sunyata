from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftminLayer(TransformLayer):
    def transform(self, x, train):
        return Z.softmin(x)


class SoftminSpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        return SoftminLayer(self.x_ndim()), form


node_wrap(SoftminSpec)
