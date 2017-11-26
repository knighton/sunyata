from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class RecurrentLayer(TransformLayer):
    def __init__(self, dim, dtype, forward=True, last=False, internal_dim=None):
        self.out_dim = dim
        self.dtype = dtype
        self.go_forward = forward
        self.ret_last = last
        self.internal_dim = internal_dim

    def params(self):
        raise NotImplementedError

    def step(self, x, prev_state, prev_internal_state):
        raise NotImplementedError

    def start_states(self, batch_size):
        shape = batch_size, self.out_dim
        initial_state = Z.constant(Z.zeros(shape, self.dtype))
        if self.internal_dim:
            shape = batch_size, self.internal_dim
            initial_internal_state = Z.constant(Z.zeros(shape, self.dtype))
        else:
            initial_internal_state = None
        return [initial_state], [initial_internal_state]

    def forward_one(self, x, is_training):
        batch_size, _, num_steps = Z.shape(x)
        states, internal_states = self.start_states(batch_size)
        steps = range(num_steps)
        if not self.go_forward:
            steps = reversed(steps)
        for step in steps:
            x_step = x[:, :, step]
            state, internal_state = self.step(
                x_step, states[-1], internal_states[-1])
            states.append(state)
            internal_states.append(state)
        if self.ret_last:
            out = states[-1]
        else:
            out = Z.stack(states[1:], 2)
        return out


class RecurrentSpec(TransformSpec):
    def __init__(self, dim=None, forward=True, last=False):
        super().__init__()
        self.out_dim = dim
        self.go_forward = forward
        self.ret_last = last

    def make_layer(self, in_dim, out_dim, dtype):
        raise NotImplementedError

    def build_one(self, form):
        in_dim, num_steps = form.shape
        out_dim = self.out_dim or in_dim
        layer = self.make_layer(in_dim, out_dim, form.dtype)
        if self.ret_last:
            out_shape = out_dim,
        else:
            out_shape = out_dim, num_steps
        form = Form(out_shape, form.dtype)
        return layer, form
