from .... import backend as Z
from ...base import node_wrap
from .base import GlobalPoolLayer, GlobalPoolSpec


class GlobalAvgPoolLayer(GlobalPoolLayer):
    def transform(self, x, train):
        return Z.global_avg_pool(x, train)


class GlobalAvgPoolSpec(GlobalPoolSpec):
    def make_layer(self, form):
        return GlobalAvgPoolLayer(self.x_ndim())


node_wrap(GlobalAvgPoolSpec, (None, 1, 2, 3))
