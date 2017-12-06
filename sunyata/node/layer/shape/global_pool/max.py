from .... import backend as Z
from .base import GlobalPoolLayer, GlobalPoolSpec


class GlobalMaxPoolLayer(GlobalPoolLayer):
    def forward_one(self, x, is_training):
        return Z.global_max_pool(x, is_training)


class GlobalMaxPoolSpec(GlobalPoolSpec):
    def make_layer(self, form):
        ndim = self.in_ndim(form.shape)
        return GlobalMaxPoolLayer(ndim)
