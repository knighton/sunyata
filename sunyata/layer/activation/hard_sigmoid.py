from ... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class HardSigmoidLayer(TransformLayer):
    def transform(self, x, train):
        return Z.hard_sigmoid(x)


class HardSigmoidSpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        return HardSigmoidLayer(self.x_ndim()), form


node_wrap(HardSigmoidSpec)
