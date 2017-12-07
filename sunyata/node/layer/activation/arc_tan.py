from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class ArcTanLayer(TransformLayer):
    def transform(self, x, train):
        return Z.arctan(x)


class ArcTanSpec(TransformSpec):
    def build_transform(self, form):
        return ArcTanLayer(self.x_ndim()), form


node_wrap(ArcTanSpec)
