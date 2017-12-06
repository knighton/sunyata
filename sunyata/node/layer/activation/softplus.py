from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftPlusLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softplus(x)


class SoftPlusSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftPlusLayer(), form


node_wrap(SoftPlusSpec)
