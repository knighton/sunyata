import numpy as np

from .. import backend as Z
from .base import Optimizer


class SGDM(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        assert 0 < lr
        assert 0 <= momentum <= 1
        self.lr = lr
        self.momentum = momentum

    def make_context(self, variable):
        arr = np.zeros(Z.shape(variable), Z.dtype_of(variable))
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'velocity': Z.cast_numpy_to(arr),
        }

    def learn(self, var, grad, ctx):
        cur_v = ctx.velocity
        new_v = -ctx.lr * grad
        ctx.velocity = cur_v * ctx.momentum + new_v * (1 - ctx.momentum)
        Z.assign(var, Z.variable_to_tensor(var) + ctx.velocity)
