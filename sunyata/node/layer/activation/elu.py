from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class ELULayer(TransformLayer):
    def __init__(self, alpha, x_ndim=None):
        super().__init__(x_ndim)
        self.alpha = alpha

    def transform(self, x, train):
        return Z.elu(x, self.alpha)


class ELUSpec(TransformSpec):
    def __init__(self, alpha=1., spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.alpha = alpha

    def build_transform(self, form):
        return ELULayer(self.alpha, self.x_ndim()), form


node_wrap(ELUSpec)
