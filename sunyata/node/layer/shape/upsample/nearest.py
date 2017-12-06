from ..... import backend as Z
from .base import UpsampleLayer, UpsampleSpec


class NearestUpsampleLayer(UpsampleLayer):
    def transform(self, x, is_training):
        return Z.nearest_upsample(x, is_training)


class NearestUpsampleSpec(UpsampleSpec):
    def make_layer(self, form):
        return NearestUpsampleLayer(self.x_ndim())
