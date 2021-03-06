from .. import backend as Z
from .base import Optimizer


class Adagrad(Optimizer):
    def __init__(self, lr=0.01, decay_rate=0.99, epsilon=1e-6):
        super().__init__()
        assert 0 < lr
        assert 0 < decay_rate < 1
        assert 0 < epsilon
        self.lr = lr
        self.decay_rate = decay_rate
        self.epsilon = epsilon

    def make_context(self, variable):
        return {
            'lr': self.lr,
            'decay_rate': self.decay_rate,
            'epsilon': self.epsilon,
            'cache': Z.zeros_like(variable),
        }

    def update_variable(self, var, grad, ctx):
        ctx.cache += Z.square(grad)
        Z.assign_sub(var, ctx.lr * grad / Z.sqrt(ctx.cache + ctx.epsilon))
