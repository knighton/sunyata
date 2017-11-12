from .. import backend as Z
from .base import Optimizer


class Adam(Optimizer):
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.99, epsilon=1e-6):
        super().__init__()
        assert 0 < lr
        assert 0 < epsilon
        assert 0 < beta1 < 1
        assert 0 < beta2 < 1
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def make_context(self, variable):
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'momentum': Z.zeros_like(variable),
            'velocity': Z.zeros_like(variable),
            'iter': 1,
        }

    def update_variable(self, var, grad, ctx):
        ctx.momentum = ctx.beta1 * ctx.momentum + (1 - ctx.beta1) * grad
        iter_momentum = ctx.momentum / (1 - Z.pow(ctx.beta1, ctx.iter))
        ctx.velocity = ctx.beta2 * ctx.velocity + \
            (1 - ctx.beta2) * Z.square(grad)
        iter_velocity = ctx.velocity / (1 - Z.pow(ctx.beta2, ctx.iter))
        ctx.iter += 1
        Z.decr(var, ctx.lr * iter_momentum /
            (Z.sqrt(iter_velocity) + ctx.epsilon))
