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
        self.reset_update_input_kernel = self.input_kernel[:, :i]
        self.reset_update_recurrent_kernel = self.recurrent_kernel[:, :i]
        self.reset_update_bias = self.bias[:i]
        self.new_input_kernel = self.input_kernel[:, i:]
        self.new_recurrent_kernel = self.recurrent_kernel[:, i:]
        self.new_bias = self.bias[i:]

    def params(self):
        return [self.input_kernel, self.recurrent_kernel, self.bias]

    def step(self, x, prev_state, prev_internal_state):
        gates = Z.sigmoid(
            Z.matmul(x, self.reset_update_input_kernel) +
            Z.matmul(prev_state, self.reset_update_recurrent_kernel) +
            self.reset_update_bias)
        i = self.out_dim
        reset_gate = gates[:, :i]
        update_gate = gates[:, i:2 * i]
        new_state = Z.tanh(
            Z.matmul(x, self.new_input_kernel) +
            Z.matmul(reset_gate * prev_state, self.new_recurrent_kernel) +
            self.new_bias)
        state = update_gate * prev_state + (1 - update_gate) * new_state
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
