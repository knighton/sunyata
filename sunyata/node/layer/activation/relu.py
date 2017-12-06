import numpy as np

from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class ReLULayer(TransformLayer):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def transform(self, x, is_training):
        return Z.relu(x, self.min, self.max)


class ReLUSpec(TransformSpec):
    def __init__(self, min=0., max=np.inf, ndim=None):
        super().__init__(ndim)
        self.min = min
        self.max = max

    def build_transform(self, form):
        return ReLULayer(self.min, self.max), form


node_wrap(ReLUSpec)
