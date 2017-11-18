from .. import backend as Z
from .base import Optimizer


class SGDM(Optimizer):
    def __init__(self, lr=0.05, momentum=0.9):
        super().__init__()
        assert 0 < lr
        assert 0 <= momentum <= 1
        self.lr = lr
        self.momentum = momentum

    def make_context(self, variable):
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'velocity': Z.zeros_like(variable),
        }

    def update_variable(self, var, grad, ctx):
        ctx.velocity = ctx.lr * grad + ctx.momentum * ctx.velocity
        Z.assign_sub(var, ctx.velocity)
