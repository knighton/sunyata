from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class RecurrentLayer(TransformLayer):
    def __init__(self, dim, dtype, forward=True, last=False):
        self.out_dim = dim
        self.dtype = dtype
        self.go_forward = forward
        self.ret_last = last

    def step(self, x, prev_state):
        raise NotImplementedError

    def make_initial_state(self, batch_size, dim):
        return Z.constant(Z.zeros((batch_size, dim), self.dtype))

    def internal_state_dim(self):
        return None

    def make_initial_internal_state(self, batch_size):
        dim = self.internal_state_dim()
        if dim is None:
            return None
        return Z.constant(Z.zeros((batch_size, dim), self.dtype))

    def forward_one(self, x, is_training):
        batch_size, _, num_steps = Z.shape(x)
        states = [self.make_initial_state(batch_size, self.out_dim)]
        internal_states = [self.make_initial_internal_state(batch_size)]
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
        self.dim = dim
        self.go_forward = forward
        self.ret_last = last

    def make_layer(self, in_dim, out_dim, dtype):
        raise NotImplementedError

    def build_one(self, form):
        in_dim, num_steps = form.shape
        out_dim = self.dim if self.dim else in_dim
        layer = self.make_layer(in_dim, out_dim, form.dtype)
        if self.ret_last:
            out_shape = out_dim,
        else:
            out_shape = out_dim, num_steps
        form = Form(out_shape, form.dtype)
        return layer, form
