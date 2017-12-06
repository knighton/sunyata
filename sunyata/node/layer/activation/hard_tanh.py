from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class HardTanhLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.hard_tanh(x)


class HardTanhSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return HardTanhLayer(), form


node_wrap(HardTanhSpec)
