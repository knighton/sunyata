from ..... import backend as Z
from .base import UpsampleLayer, UpsampleSpec


class LinearUpsampleLayer(UpsampleLayer):
    def transform(self, x, is_training):
        return Z.linear_upsample(x, is_training)


class LinearUpsampleSpec(UpsampleSpec):
    def make_layer(self, form):
        return LinearUpsampleLayer(self.x_ndim())
