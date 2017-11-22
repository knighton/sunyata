from .... import backend as Z
from .base import PoolLayer, PoolSpec


class MaxPoolLayer(PoolLayer):
    def forward_one(self, x, is_training):
        return Z.max_pool(x, self.face, self.stride, self.pad)


class MaxPoolSpec(PoolSpec):
    def make_layer(self, form):
        ndim = self.in_ndim(form.shape)
        return MaxPoolLayer(self.face, self.stride, self.pad, ndim)
