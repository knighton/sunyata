from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class HardShrinkLayer(TransformLayer):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def transform(self, x, is_training):
        return Z.hard_shrink(x, self.lam)


class HardShrinkSpec(TransformSpec):
    def __init__(self, lam=0.5, ndim=None):
        super().__init__(ndim)
        self.lam = lam

    def build_transform(self, form):
        return HardShrinkLayer(self.lam), form


node_wrap(HardShrinkSpec)
