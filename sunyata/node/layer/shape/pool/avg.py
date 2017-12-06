from .... import backend as Z
from .base import PoolLayer, PoolSpec


class AvgPoolLayer(PoolLayer):
    def forward_one(self, x, is_training):
        return Z.avg_pool(x, self.face, self.stride, self.pad)


class AvgPoolSpec(PoolSpec):
    def make_layer(self, form):
        ndim = self.in_ndim(form.shape)
        return AvgPoolLayer(self.face, self.stride, self.pad, ndim)
