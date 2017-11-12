import numpy as np

from .. import backend as Z
from .base import Form, TransformLayer, TransformSpec


class DenseLayer(TransformLayer):
    def __init__(self, kernel, bias):
        if Z.name == 'chainer':
            kernel = kernel.T
        self.kernel = Z.variable(Z.cast_numpy_to(kernel))
        self.bias = Z.variable(Z.cast_numpy_to(bias))

    def params(self):
        return [self.kernel, self.bias]

    def forward_one(self, x):
        return Z.dense(x, self.kernel, self.bias)


class DenseSpec(TransformSpec):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def build_one(self, form):
        assert len(form.shape) == 1
        in_dim, = form.shape
        kernel = np.random.normal(
            0, 0.1, (in_dim, self.out_dim)).astype('float32')
        bias = np.random.normal(0, 0.1, (self.out_dim,)).astype('float32')
        out_shape = self.out_dim,
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)
