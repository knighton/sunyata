from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class IdentityLayer(TransformLayer):
    def transform(self, x, train):
        return Z.identity(x)


class IdentitySpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        return IdentityLayer(self.x_ndim()), form


node_wrap(IdentitySpec)
