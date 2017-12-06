from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SigmoidLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.sigmoid(x)


class SigmoidSpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        return SigmoidLayer(self.x_ndim()), form


node_wrap(SigmoidSpec)
