from ..... import backend as Z
from .base import UpsampleLayer, UpsampleSpec


class NearestUpsampleLayer(UpsampleLayer):
    def transform(self, x, train):
        return Z.nearest_upsample(x, train)


class NearestUpsampleSpec(UpsampleSpec):
    def make_layer(self, form):
        return NearestUpsampleLayer(self.x_ndim())
