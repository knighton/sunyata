from .... import backend as Z
from .... import init
from .base import RecurrentLayer, RecurrentSpec


class ElmanRULayer(RecurrentLayer):
    def __init__(self, forward, last, input_kernel, recurrent_kernel, bias):
        dim = input_kernel.shape[1]
        dtype = input_kernel.dtype.name
        super().__init__(dim, dtype, forward, last)
        self.input_kernel = self.add_param(input_kernel)
        self.recurrent_kernel = self.add_param(recurrent_kernel)
        self.bias = self.add_param(bias)

    def step(self, x, prev_state, prev_internal_state):
        return Z.tanh(Z.matmul(x, self.input_kernel) +
                      Z.matmul(prev_state, self.recurrent_kernel) +
                      self.bias), None


class ElmanRUSpec(RecurrentSpec):
    def __init__(self, dim=None, forward=True, last=False,
                 input_kernel_init='glorot_uniform',
                 recurrent_kernel_init='orthogonal', bias_init='zeros'):
        super().__init__(dim, forward, last)
        self.input_kernel_init = init.get(input_kernel_init)
        self.recurrent_kernel_init = init.get(recurrent_kernel_init)
        self.bias_init = init.get(bias_init)

    def make_layer(self, in_dim, out_dim, dtype):
        input_kernel_shape = in_dim, out_dim
        input_kernel = self.input_kernel_init.get(input_kernel_shape, dtype)
        recurrent_kernel_shape = out_dim, out_dim
        recurrent_kernel = self.recurrent_kernel_init.get(
            recurrent_kernel_shape, dtype)
        bias_shape = out_dim,
        bias = self.bias_init.get(bias_shape, dtype)
        return ElmanRULayer(self.go_forward, self.ret_last, input_kernel,
                            recurrent_kernel, bias)
