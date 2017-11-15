from ... import backend as Z
from ..base import TransformLayer, TransformSpec


class ReLULayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.relu(x)


class ReLUSpec(TransformSpec):
    def build_one(self, form):
        return ReLULayer(), form
