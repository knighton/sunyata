import numpy as np

from ... import backend as Z
from .base import RecurrentLayer, RecurrentSpec


class SimpleRULayer(RecurrentLayer):
    def __init__(self, forward, last, input_kernel, recurrent_kernel, bias):
        dim = input_kernel.shape[1]
        dtype = input_kernel.dtype.name
        super().__init__(dim, dtype, forward, last)
        self.input_kernel = Z.variable(Z.numpy_to_device(input_kernel))
        self.recurrent_kernel = Z.variable(Z.numpy_to_device(recurrent_kernel))
        self.bias = Z.variable(Z.numpy_to_device(bias))

    def params(self):
        return [self.input_kernel, self.recurrent_kernel, self.bias]

    def step(self, x, prev_state, prev_internal_state):
        return Z.tanh(Z.matmul(x, self.input_kernel) +
                      Z.matmul(prev_state, self.recurrent_kernel) +
                      self.bias), None


class SimpleRUSpec(RecurrentSpec):
    def __init__(self, dim=None, forward=True, last=False):
        super().__init__(dim, forward, last)

    def make_layer(self, in_dim, out_dim, dtype):
        input_kernel_shape = in_dim, out_dim
        input_kernel = np.random.normal(
            0, 0.1, input_kernel_shape).astype(dtype)
        recurrent_kernel_shape = out_dim, out_dim
        recurrent_kernel = np.random.normal(
            0, 0.1, recurrent_kernel_shape).astype(dtype)
        bias_shape = out_dim,
        bias = np.random.normal(0, 0.1, bias_shape).astype(dtype)
        return SimpleRULayer(self.go_forward, self.ret_last, input_kernel,
                             recurrent_kernel, bias)
