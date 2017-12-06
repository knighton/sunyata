from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class LeakyReLULayer(TransformLayer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def transform(self, x, is_training):
        return Z.leaky_relu(x, self.alpha)


class LeakyReLUSpec(TransformSpec):
    def __init__(self, alpha=0.1, ndim=None):
        super().__init__(ndim)
        self.alpha = alpha

    def build_transform(self, form):
        return LeakyReLULayer(self.alpha), form


node_wrap(LeakyReLUSpec)
