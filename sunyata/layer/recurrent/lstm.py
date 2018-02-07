from ... import backend as Z
from ... import init
from ..base import LinkBuilder
from .base import RecurrentLayer, RecurrentSpec


class LSTMLayer(RecurrentLayer):
    def __init__(self, forward, last, input_kernel, recurrent_kernel, bias):
        dim = input_kernel.shape[1] // 4
        dtype = input_kernel.dtype.name
        super().__init__(dim, dtype, forward, last, dim)
        self.input_kernel = self.add_param(input_kernel)
        self.recurrent_kernel = self.add_param(recurrent_kernel)
        self.bias = self.add_param(bias)

    def step(self, x, prev_state, prev_internal_state):
        a = Z.matmul(x, self.input_kernel) + \
            Z.matmul(prev_state, self.recurrent_kernel) + self.bias
        index = self.out_dim
        i = Z.sigmoid(a[:, :index])
        f = Z.sigmoid(a[:, index:2 * index])
        o = Z.sigmoid(a[:, 2 * index:3 * index])
        g = Z.tanh(a[:, 3 * index:])
        next_internal_state = f * prev_internal_state + i * g
        next_state = o * Z.tanh(next_internal_state)
        return next_state, next_internal_state


class LSTMSpec(RecurrentSpec):
    def __init__(self, dim=None, forward=True, last=False,
                 input_kernel_init='glorot_uniform',
                 recurrent_kernel_init='orthogonal', bias_init='zeros'):
        super().__init__(dim, forward, last)
        self.input_kernel_init = init.get(input_kernel_init)
        self.recurrent_kernel_init = init.get(recurrent_kernel_init)
        self.bias_init = init.get(bias_init)

    def make_layer(self, in_dim, out_dim, dtype):
        input_kernel_shape = in_dim, 4 * out_dim
        input_kernel = self.input_kernel_init(
            input_kernel_shape, dtype, 'conv_kernel')
        recurrent_kernel_shape = out_dim, 4 * out_dim
        recurrent_kernel = self.recurrent_kernel_init(
            recurrent_kernel_shape, dtype)
        bias_shape = 4 * out_dim,
        bias = self.bias_init(bias_shape, dtype)
        return LSTMLayer(self.go_forward, self.ret_last, input_kernel,
                         recurrent_kernel, bias)


LSTM = LinkBuilder(LSTMSpec)
