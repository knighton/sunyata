from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SELULayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.selu(x)


class SELUSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SELULayer(), form


node_wrap(SELUSpec)
