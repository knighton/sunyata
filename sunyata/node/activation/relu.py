from ... import backend as Z
from ..base import TransformLayer, TransformSpec


class ReLULayer(TransformLayer):
    def forward_one(self, x):
        return Z.clip(x, min=0)


class ReLUSpec(TransformSpec):
    def build_one(self, form):
        return ReLULayer(), form
