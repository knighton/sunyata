from .. import backend as Z
from .base import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        assert 0 < lr
        self.lr = lr

    def make_context(self, variable):
        return {'lr': self.lr}

    def learn(self, var, grad, ctx):
        Z.assign(var, Z.variable_to_tensor(var) - ctx.lr * grad)
