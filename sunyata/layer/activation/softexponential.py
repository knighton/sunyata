from ... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftExponentialLayer(TransformLayer):
    def transform(self, x, train):
        return Z.softexponential(x)


class SoftExponentialSpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        return SoftExponentialLayer(self.x_ndim()), form


node_wrap(SoftExponentialSpec)
