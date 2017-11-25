import numpy as np

from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class SimpleRNNLayer(TransformLayer):
    def __init__(self, input_kernel, recurrent_kernel, bias, go, ret):
        self.input_kernel = Z.variable(Z.numpy_to_device(input_kernel))
        self.recurrent_kernel = Z.variable(Z.numpy_to_device(recurrent_kernel))
        self.bias = Z.variable(Z.numpy_to_device(bias))
        self.out_dim = Z.shape(self.input_kernel)[1]
        self.go = go
        self.ret = ret

    def step(self, x, prev_state):
        return Z.tanh(Z.matmul(x, self.input_kernel) +
                      Z.matmul(prev_state, self.recurrent_kernel) +
                      self.bias)

    def forward_one(self, x, is_training):
        batch_size, _, num_timesteps = Z.shape(x)
        initial_state_shape = batch_size, self.out_dim
        initial_state = Z.constant(Z.zeros(initial_state_shape, Z.dtype_of(x)))
        states = [initial_state]
        timesteps = range(num_timesteps)
        if self.go == 'backward':
            timesteps = reversed(timesteps)
        for timestep in timesteps:
            x_step = x[:, :, timestep]
            next_state = self.step(x_step, states[-1])
            states.append(next_state)
        if self.ret == 'all':
            out = Z.stack(states[1:], 2)
        elif self.ret == 'last':
            out = states[-1]
        else:
            assert False
        return out


class SimpleRNNSpec(TransformSpec):
    def __init__(self, dim=None, go='forward', ret='all'):
        super().__init__()
        self.dim = dim
        self.go = go
        self.ret = ret

    def build_one(self, form):
        in_dim, num_timesteps = form.shape
        out_dim = self.dim if self.dim else in_dim
        input_kernel_shape = in_dim, out_dim
        input_kernel = np.random.normal(
            0, 0.1, input_kernel_shape).astype(form.dtype)
        recurrent_kernel_shape = out_dim, out_dim
        recurrent_kernel = np.random.normal(
            0, 0.1, recurrent_kernel_shape).astype(form.dtype)
        bias_shape = out_dim,
        bias = np.random.normal(0, 0.1, bias_shape).astype(form.dtype)
        layer = SimpleRNNLayer(input_kernel, recurrent_kernel, bias, self.go,
                               self.ret)
        if self.ret == 'all':
            out_shape = out_dim, num_timesteps
        elif self.ret == 'last':
            out_shape = out_dim,
        else:
            assert False
        form = Form(out_shape, form.dtype)
        return layer, form
