from ..base import APIMixin


class BaseReduceAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def argmin(self, axis=-1):
        raise NotImplementedError

    def argmax(self, axis=-1):
        raise NotImplementedError

    def min(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def max(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def mean(self, x, axis=None, keepdims=False):
        old_size = self.size(x)
        x = self.sum(x, axis, keepdims)
        ratio = old_size / self.size(x)
        return x / ratio

    def prod(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def var(self, x, axis=None, keepdims=False):
        mean = self.mean(x, axis, keepdims)
        var = self.square(x - mean)
        return self.mean(var, axis, keepdims)

    def std(self, x, axis=None, keepdims=False):
        return self.sqrt(self.var(x, axis, keepdims))

    def moments(self, x, axis=None, keepdims=False):
        shift = self.mean(x, axis, True)
        shifted = x - shift
        shifted_mean = self.mean(shifted, axis, True)
        var_mean = self.mean(self.square(shifted), axis, True)
        var = var_mean - self.square(shifted_mean)
        mean = shifted_mean + shift
        return mean, var

    def any(self, x, axis=None, keepdims=False, dtype=None):
        nonneg = self.abs(x)
        minima = self.min(nonneg, axis, keepdims)
        return self.less(0, minima, dtype)

    def all(self, x, axis=None, keepdims=False, dtype=None):
        nonneg = self.abs(x)
        nonzero = self.less(0, nonneg, 'int64')
        sums = self.sum(nonzero, axis, keepdims)
        return self.less(0, sums, dtype)
