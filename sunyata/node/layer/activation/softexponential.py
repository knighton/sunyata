from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftExponentialLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softexponential(x)


class SoftExponentialSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftExponentialLayer(), form


node_wrap(SoftExponentialSpec)
