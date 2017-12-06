from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class TanhLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.tanh(x)


class TanhSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return TanhLayer(), form


node_wrap(TanhSpec)
