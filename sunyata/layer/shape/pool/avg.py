from .... import backend as Z
from ...base import node_wrap
from .base import PoolLayer, PoolSpec


class AvgPoolLayer(PoolLayer):
    def transform(self, x, train):
        return Z.avg_pool(x, self.face, self.stride, self.pad)


class AvgPoolSpec(PoolSpec):
    def make_layer(self, form):
        return AvgPoolLayer(self.face, self.stride, self.pad, self.x_ndim())


node_wrap(AvgPoolSpec)
