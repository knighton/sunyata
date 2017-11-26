import numpy as np

from ... import backend as Z
from .base import RecurrentLayer, RecurrentSpec


class GRULayer(RecurrentLayer):
    def __init__(self, forward, last, input_kernel, recurrent_kernel, bias):
        dim = input_kernel.shape[1] // 3
        dtype = input_kernel.dtype.name
        super().__init__(dim, dtype, forward, last)
        self.input_kernel = Z.variable(Z.numpy_to_device(input_kernel))
        self.recurrent_kernel = Z.variable(Z.numpy_to_device(recurrent_kernel))
        self.bias = Z.variable(Z.numpy_to_device(bias))
        i = 2 * self.out_dim
        self.rz_input_kernel = self.input_kernel[:, :i]
        self.rz_recurrent_kernel = self.recurrent_kernel[:, :i]
        self.rz_bias = self.bias[:i]
        self.h_input_kernel = self.input_kernel[:, i:]
        self.h_recurrent_kernel = self.recurrent_kernel[:, i:]
        self.h_bias = self.bias[i:]

    def step(self, x, prev_state, prev_internal_state):
        rz = Z.sigmoid(Z.matmul(x, self.rz_input_kernel) +
                       Z.matmul(prev_state, self.rz_recurrent_kernel) +
                       self.rz_bias)
        r = rz[:, :self.out_dim]
        z = rz[:, self.out_dim:2 * self.out_dim]
        h = Z.tanh(Z.matmul(x, self.h_input_kernel) +
                   Z.matmul(r * prev_state, self.h_recurrent_kernel) +
                   self.h_bias)
        state = z * prev_state + (1 - z) * h
        return state, None


class GRUSpec(RecurrentSpec):
    def __init__(self, dim=None, forward=True, last=False):
        super().__init__(dim, forward, last)

    def make_layer(self, in_dim, out_dim, dtype):
        input_kernel_shape = in_dim, 3 * out_dim
        input_kernel = np.random.normal(
            0, 0.1, input_kernel_shape).astype(dtype)
        recurrent_kernel_shape = out_dim, 3 * out_dim
        recurrent_kernel = np.random.normal(
            0, 0.1, recurrent_kernel_shape).astype(dtype)
        bias_shape = 3 * out_dim,
        bias = np.random.normal(0, 0.1, bias_shape).astype(dtype)
        return GRULayer(self.go_forward, self.ret_last, input_kernel,
                        recurrent_kernel, bias)
