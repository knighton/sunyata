from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class SELULayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.selu(x)


class SELUSpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        return SELULayer(self.x_ndim()), form


node_wrap(SELUSpec)
