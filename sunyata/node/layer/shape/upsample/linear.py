from ..... import backend as Z
from .base import UpsampleLayer, UpsampleSpec


class LinearUpsampleLayer(UpsampleLayer):
    def transform(self, x, train):
        return Z.linear_upsample(x, train)


class LinearUpsampleSpec(UpsampleSpec):
    def make_layer(self, form):
        return LinearUpsampleLayer(self.x_ndim())
