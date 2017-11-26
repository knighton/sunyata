from ... import backend as Z
from ... import init
from .base import RecurrentLayer, RecurrentSpec


class MinimalRULayer(RecurrentLayer):
    def __init__(self, forward, last, gate_input_kernel, gate_recurrent_kernel,
                 gate_bias, latent_kernel, latent_bias):
        dim = gate_input_kernel.shape[1]
        dtype = gate_input_kernel.dtype.name
        super().__init__(dim, dtype, forward, last)
        self.gate_input_kernel = self.add_param(gate_input_kernel)
        self.gate_recurrent_kernel = self.add_param(gate_recurrent_kernel)
        self.gate_bias = self.add_param(gate_bias)
        self.latent_kernel = self.add_param(latent_kernel)
        self.latent_bias = self.add_param(latent_bias)

    def step(self, x, prev_state, prev_internal_state):
        latent_x = Z.tanh(Z.dense(x, self.latent_kernel, self.latent_bias))
        update_gate = Z.sigmoid(
            Z.matmul(latent_x, self.gate_input_kernel) +
            Z.matmul(prev_state, self.gate_recurrent_kernel) + self.gate_bias)
        state = update_gate * prev_state + (1 - update_gate) * latent_x
        return state, None


class MinimalRUSpec(RecurrentSpec):
    def __init__(self, dim=None, forward=True, last=False,
                 gate_input_kernel_init='glorot_uniform',
                 gate_recurrent_kernel_init='orthogonal',
                 gate_bias_init='zeros', latent_kernel_init='glorot_uniform',
                 latent_bias_init='zeros'):
        super().__init__(dim, forward, last)
        self.gate_input_kernel_init = init.get(gate_input_kernel_init)
        self.gate_recurrent_kernel_init = init.get(gate_recurrent_kernel_init)
        self.gate_bias_init = init.get(gate_bias_init)
        self.latent_kernel_init = init.get(latent_kernel_init)
        self.latent_bias_init = init.get(latent_bias_init)

    def make_layer(self, in_dim, out_dim, dtype):
        gate_input_kernel_shape = out_dim, out_dim
        gate_input_kernel = self.gate_input_kernel_init(
            gate_input_kernel_shape, dtype)
        gate_recurrent_kernel_shape = out_dim, out_dim
        gate_recurrent_kernel = self.gate_recurrent_kernel_init(
            gate_recurrent_kernel_shape, dtype)
        gate_bias_shape = out_dim,
        gate_bias = self.gate_bias_init(gate_bias_shape, dtype)
        latent_kernel_shape = in_dim, out_dim
        latent_kernel = self.latent_kernel_init(
            latent_kernel_shape, dtype, 'conv_kernel')
        latent_bias_shape = out_dim,
        latent_bias = self.latent_bias_init(latent_bias_shape, dtype)
        return MinimalRULayer(
            self.go_forward, self.ret_last, gate_input_kernel,
            gate_recurrent_kernel, gate_bias, latent_kernel, latent_bias)
