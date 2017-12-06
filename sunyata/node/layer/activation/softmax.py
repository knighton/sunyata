from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftmaxLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softmax(x)


class SoftmaxSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftmaxLayer(), form


node_wrap(SoftmaxSpec)
