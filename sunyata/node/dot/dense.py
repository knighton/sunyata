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
        params = [self.kernel]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def forward_one(self, x, is_training):
        return Z.dense(x, self.kernel, self.bias)


class DenseSpec(TransformSpec):
    def __init__(self, dim=None, has_bias=True):
        self.out_dim = dim
        self.has_bias = has_bias

    def build_one(self, form):
        assert len(form.shape) == 1
        in_dim, = form.shape
        out_dim = in_dim if self.out_dim is None else self.out_dim
        kernel_shape = in_dim, out_dim
        kernel = np.random.normal(0, 0.1, kernel_shape).astype(form.dtype)
        if self.has_bias:
            bias_shape = out_dim,
            bias = np.zeros(bias_shape, form.dtype)
        else:
            bias = None
        out_shape = out_dim,
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)
