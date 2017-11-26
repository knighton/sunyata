import numpy as np

from ... import backend as Z
from .base import RecurrentLayer, RecurrentSpec


class MinimalRNNLayer(RecurrentLayer):
    def __init__(self, forward, last, gate_input_kernel, gate_recurrent_kernel,
                 gate_bias, latent_kernel, latent_bias):
        dim = gate_input_kernel.shape[1]
        dtype = gate_input_kernel.dtype.name
        super().__init__(dim, dtype, forward, last)
        self.gate_input_kernel = \
            Z.variable(Z.numpy_to_device(gate_input_kernel))
        self.gate_recurrent_kernel = \
            Z.variable(Z.numpy_to_device(gate_recurrent_kernel))
        self.gate_bias = Z.variable(Z.numpy_to_device(gate_bias))
        self.latent_kernel = Z.variable(Z.numpy_to_device(latent_kernel))
        self.latent_bias = Z.variable(Z.numpy_to_device(latent_bias))

    def params(self):
        return [self.gate_input_kernel, self.gate_recurrent_kernel,
                self.gate_bias, self.latent_kernel, self.latent_bias]

    def step(self, x, prev_state, prev_internal_state):
        latent_x = Z.tanh(Z.dense(x, self.latent_kernel, self.latent_bias))
        update_gate = Z.sigmoid(
            Z.matmul(latent_x, self.gate_input_kernel) +
            Z.matmul(prev_state, self.gate_recurrent_kernel) + self.gate_bias)
        state = update_gate * prev_state + (1 - update_gate) * latent_x
        return state, None


class MinimalRNNSpec(RecurrentSpec):
    def __init__(self, dim=None, forward=True, last=False):
        super().__init__(dim, forward, last)

    def make_layer(self, in_dim, out_dim, dtype):
        gate_input_kernel_shape = out_dim, out_dim
        gate_input_kernel = np.random.normal(
            0, 0.1, gate_input_kernel_shape).astype(dtype)
        gate_recurrent_kernel_shape = out_dim, out_dim
        gate_recurrent_kernel = np.random.normal(
            0, 0.1, gate_recurrent_kernel_shape).astype(dtype)
        gate_bias_shape = out_dim,
        gate_bias = np.random.normal(0, 0.1, gate_bias_shape).astype(dtype)
        latent_kernel_shape = in_dim, out_dim
        latent_kernel = np.random.normal(
            0, 0.1, latent_kernel_shape).astype(dtype)
        latent_bias_shape = out_dim,
        latent_bias = np.zeros(latent_bias_shape, dtype)
        return MinimalRNNLayer(
            self.go_forward, self.ret_last, gate_input_kernel,
            gate_recurrent_kernel, gate_bias, latent_kernel, latent_bias)
