from .. import backend as Z
from .base import Optimizer


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def update_param(self, gradient, variable):
        Z.assign(variable, Z.variable_to_tensor(variable) - self.lr * gradient)
