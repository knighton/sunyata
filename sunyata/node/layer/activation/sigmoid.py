from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SigmoidLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.sigmoid(x)


class SigmoidSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SigmoidLayer(), form


node_wrap(SigmoidSpec)
