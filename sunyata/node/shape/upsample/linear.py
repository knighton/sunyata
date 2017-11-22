from .... import backend as Z
from .base import UpsampleLayer, UpsampleSpec


class LinearUpsampleLayer(UpsampleLayer):
    def forward_one(self, x, is_training):
        return Z.linear_upsample(x, is_training)


class LinearUpsampleSpec(UpsampleSpec):
    def make_layer(self, form):
        ndim = self.in_ndim(form.shape)
        return LinearUpsampleLayer(ndim)
