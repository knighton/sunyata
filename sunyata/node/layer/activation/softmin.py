from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftminLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softmin(x)


class SoftminSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftminLayer(), form


node_wrap(SoftminSpec)
