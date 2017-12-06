from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class ELULayer(TransformLayer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def transform(self, x, is_training):
        return Z.elu(x, self.alpha)


class ELUSpec(TransformSpec):
    def __init__(self, alpha=1., ndim=None):
        super().__init__(ndim)
        self.alpha = alpha

    def build_transform(self, form):
        return ELULayer(self.alpha), form


node_wrap(ELUSpec)
