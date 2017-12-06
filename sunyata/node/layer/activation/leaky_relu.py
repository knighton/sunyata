from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class LeakyReLULayer(TransformLayer):
    def __init__(self, alpha, x_ndim=None):
        super().__init__(x_ndim)
        self.alpha = alpha

    def transform(self, x, is_training):
        return Z.leaky_relu(x, self.alpha)


class LeakyReLUSpec(TransformSpec):
    def __init__(self, alpha=0.1, spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.alpha = alpha

    def build_transform(self, form):
        return LeakyReLULayer(self.alpha, self.x_ndim()), form


node_wrap(LeakyReLUSpec)
