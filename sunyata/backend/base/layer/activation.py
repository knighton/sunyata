import numpy as np

from ..base import APIMixin


class BaseActivationAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)
        self._selu_alpha = 1.6732632423543772848170429916717
        self._selu_scale = 1.0507009873554804934193349852946

    def arctan_dx(self, x):
        return 1 / (self.square(x) + 1)

    def bent_identity(self, x):
        return (self.sqrt(self.square(x) + 1) - 1) / 2 + x

    def bent_identity_dx(self, x):
        return x / (2 * self.sqrt(self.square(x) + 1)) + 1

    def elu(self, x, alpha=1):
        return self.where(self.less(x, 0), alpha * self.expm1(x), x)

    def elu_dx(self, x, alpha=1):
        return self.where(self.less(x, 0), self.elu(x, alpha) + alpha, 1)

    def hard_shrink(self, x, lam=0.5):
        return (self.less(x, -lam) + self.less(lam, x)) * x

    def hard_shrink_dx(self, x, lam=0.5):
        return self.less(x, -lam) + self.less(lam, x)

    def hard_sigmoid(self, x):
        return self.clip(0.2 * x + 0.5, 0, 1)

    def hard_sigmoid_dx(self, x):
        return 0.2 * self.less(-2.5, x) * self.less(x, 2.5)

    def hard_tanh(self, x):
        return self.clip(x, -1, 1)

    def hard_tanh_dx(self, x):
        return self.less(-1, x) * self.less(x, 1)

    def identity(self, x):
        return x

    def identity_dx(self, x):
        return 0 * x + 1

    def leaky_relu(self, x, alpha=0.1):
        x = self.relu(x)
        if alpha != 0:
            x -= alpha * self.relu(-x)
        return x

    def leaky_relu_dx(self, x, alpha=0.1):
        return self.where(self.less(x, 0), alpha, 1)

    def log_sigmoid(self, x):
        return self.log(self.sigmoid(x))

    def log_sigmoid_dx(self, x):
        return 1 / (self.exp(x) + 1)

    def log_softmax(self, x):
        return self.log(self.softmax(x))

    def relu(self, x, min=0, max=np.inf):
        return self.clip(x, min, max)

    def relu_dx(self, x, min=0, max=np.inf):
        return self.less(min, x) * self.less(x, max)

    def selu(self, x):
        return self._selu_scale * self.elu(x, self._selu_alpha)

    def selu_dx(self, x):
        return self._selu_scale * self.elu_dx(x, self._selu_alpha)

    def sigmoid(self, x):
        return 1 / (self.exp(-x) + 1)

    def sigmoid_dx(self, x):
        return self.exp(x) / self.square(self.exp(x) + 1)

    def softexponential(self, x, alpha=0.25):
        if alpha < 0:
            x = -self.log(1 - alpha * (x + alpha)) / alpha
        elif alpha == 0:
            pass
        else:
            x = self.expm1(alpha * x) / alpha + alpha
        return x

    def softexponential_dx(self, x, alpha=0.25):
        if alpha < 0:
            x = 1 / (1 - alpha * (x + alpha))
        elif alpha == 0:
            x = 1
        else:
            x = self.exp(alpha * x)
        return x

    def softmax(self, x):
        axes = list(range(self.ndim(x)))[1:]
        e_x = self.exp(x)
        return e_x / self.sum(e_x, axes, True)

    def softmin(self, x):
        axes = list(range(self.ndim(x)))[1:]
        maxes = self.max(x, axes, True)
        e_x = self.exp(-maxes - x)
        return e_x / self.sum(e_x, axes, True)

    def softplus(self, x, beta=1, threshold=20):
        curve = 1 / beta * self.log1p(self.exp(beta * x))
        return self.where(self.less(x, threshold), curve, x)

    def softplus_dx(self, x, beta=1, threshold=20):
        e = self.exp(beta * x)
        curve = e / (e + 1)
        return self.where(self.less(x, threshold), curve, 1)

    def softshrink(self, x, lam=0.5):
        return self.sign(x) * self.maximum(self.abs(x) - lam, 0)

    def softshrink_dx(self, x, lam=0.5):
        return self.less(x, -lam) + self.less(lam, x)

    def softsign(self, x):
        return x / (self.abs(x) + 1)

    def softsign_dx(self, x):
        return 1 / self.square(self.abs(x) + 1)

    def tanh_dx(self, x):
        return 4 / self.square(self.exp(-x) + self.exp(x))

    def tanh_shrink(self, x):
        return x - self.tanh(x)

    def tanh_shrink_dx(self, x):
        num = self.expm1(2 * x)
        denom = self.exp(2 * x) + 1
        return self.square(num) / self.square(denom)
