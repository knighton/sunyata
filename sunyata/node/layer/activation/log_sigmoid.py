from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class LogSigmoidLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.log_sigmoid(x)


class LogSigmoidSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return LogSigmoidLayer(), form


node_wrap(LogSigmoidSpec)
