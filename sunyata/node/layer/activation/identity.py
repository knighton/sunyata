from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class IdentityLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.identity(x)


class IdentitySpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return IdentityLayer(), form


node_wrap(IdentitySpec)
