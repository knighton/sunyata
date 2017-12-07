from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SoftmaxLayer(TransformLayer):
    def transform(self, x, train):
        return Z.softmax(x)


class SoftmaxSpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        return SoftmaxLayer(self.x_ndim()), form


node_wrap(SoftmaxSpec)
