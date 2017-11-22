import numpy as np

from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class DenseLayer(TransformLayer):
    def __init__(self, kernel, bias):
        if Z.name == 'chainer':
            kernel = kernel.T
        self.kernel = Z.variable(Z.numpy_to_device(kernel))
        self.bias = Z.variable(Z.numpy_to_device(bias))

    def params(self):
        return [self.kernel, self.bias]

    def forward_one(self, x, is_training):
        return Z.dense(x, self.kernel, self.bias)


class DenseSpec(TransformSpec):
    def __init__(self, dim):
        self.out_dim = dim

    def build_one(self, form):
        assert len(form.shape) == 1
        in_dim, = form.shape
        kernel_shape = in_dim, self.out_dim
        kernel = np.random.normal(0, 0.1, kernel_shape).astype(form.dtype)
        bias_shape = self.out_dim,
        bias = np.zeros(bias_shape, form.dtype)
        out_shape = bias_shape
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)
