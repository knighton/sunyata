from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class ArcTanLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.arctan(x)


class ArcTanSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return ArcTanLayer(), form


node_wrap(ArcTanSpec)
