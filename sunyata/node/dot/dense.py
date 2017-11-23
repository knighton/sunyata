import numpy as np

from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class DenseLayer(TransformLayer):
    def __init__(self, kernel, bias):
        if Z.name == 'chainer':
            kernel = kernel.T
        self.kernel = Z.variable(Z.numpy_to_device(kernel))
        if bias is None:
            self.bias = None
        else:
            self.bias = Z.variable(Z.numpy_to_device(bias))

    def params(self):
        if self.bias is None:
            variables = [self.kernel]
        else:
            variables = [self.kernel, self.bias]
        return variables

    def forward_one(self, x, is_training):
        return Z.dense(x, self.kernel, self.bias)


class DenseSpec(TransformSpec):
    def __init__(self, dim, has_bias=True):
        self.out_dim = dim
        self.has_bias = has_bias

    def build_one(self, form):
        assert len(form.shape) == 1
        in_dim, = form.shape
        kernel_shape = in_dim, self.out_dim
        kernel = np.random.normal(0, 0.1, kernel_shape).astype(form.dtype)
        if self.has_bias:
            bias_shape = self.out_dim,
            bias = np.zeros(bias_shape, form.dtype)
        else:
            bias = None
        out_shape = self.out_dim,
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)
