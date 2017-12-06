from .... import backend as Z
from .base import GlobalPoolLayer, GlobalPoolSpec


class GlobalAvgPoolLayer(GlobalPoolLayer):
    def forward_one(self, x, is_training):
        return Z.global_avg_pool(x, is_training)


class GlobalAvgPoolSpec(GlobalPoolSpec):
    def make_layer(self, form):
        ndim = self.in_ndim(form.shape)
        return GlobalAvgPoolLayer(ndim)
