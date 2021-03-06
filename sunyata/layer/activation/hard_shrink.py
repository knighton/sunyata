from ... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class HardShrinkLayer(TransformLayer):
    def __init__(self, lam, x_ndim=None):
        super().__init__(x_ndim)
        self.lam = lam

    def transform(self, x, train):
        return Z.hard_shrink(x, self.lam)


class HardShrinkSpec(TransformSpec):
    def __init__(self, lam=0.5, spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.lam = lam

    def build_transform(self, form):
        return HardShrinkLayer(self.lam, self.x_ndim()), form


node_wrap(HardShrinkSpec)
