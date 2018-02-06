from .... import backend as Z
from ...base import node_wrap
from .base import GlobalPoolLayer, GlobalPoolSpec


class GlobalMaxPoolLayer(GlobalPoolLayer):
    def transform(self, x, train):
        return Z.global_max_pool(x, train)


class GlobalMaxPoolSpec(GlobalPoolSpec):
    def make_layer(self, form):
        return GlobalMaxPoolLayer(self.x_ndim())


node_wrap(GlobalMaxPoolSpec, (None, 1, 2, 3))
