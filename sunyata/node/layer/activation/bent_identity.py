from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class BentIdentityLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.bent_identity(x)


class BentIdentitySpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return BentIdentityLayer(), form


node_wrap(BentIdentitySpec)
