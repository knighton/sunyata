from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftSignLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softsign(x)


class SoftSignSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftSignLayer(), form


node_wrap(SoftSignSpec)
