from .... import backend as Z
from .base import UpsampleLayer, UpsampleSpec


class NearestUpsampleLayer(UpsampleLayer):
    def forward_one(self, x, is_training):
        return Z.nearest_upsample(x, is_training)


class NearestUpsampleSpec(UpsampleSpec):
    def make_layer(self, form):
        ndim = self.in_ndim(form.shape)
        return NearestUpsampleLayer(ndim)
