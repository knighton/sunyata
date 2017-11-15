from ... import backend as Z
from ..base import TransformLayer, TransformSpec


class SoftmaxLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softmax(x)


class SoftmaxSpec(TransformSpec):
    def build_one(self, form):
        return SoftmaxLayer(), form
