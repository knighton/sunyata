from .. import backend as Z
from .base import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.05):
        super().__init__()
        assert 0 < lr
        self.lr = lr

    def make_context(self, variable):
        return {'lr': self.lr}

    def update_variable(self, var, grad, ctx):
        Z.decr(var, ctx.lr * grad)
