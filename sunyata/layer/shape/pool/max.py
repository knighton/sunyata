from .... import backend as Z
from ...base import node_wrap
from .base import PoolLayer, PoolSpec


class MaxPoolLayer(PoolLayer):
    def transform(self, x, train):
        return Z.max_pool(x, self.face, self.stride, self.pad)


class MaxPoolSpec(PoolSpec):
    def make_layer(self, form):
        return MaxPoolLayer(self.face, self.stride, self.pad, self.x_ndim())


node_wrap(MaxPoolSpec)
