from .base import Transformer


class Wrap(Transformer):
    def __init__(self, func, inverse=None):
        self.func = func
        self.inverse = inverse

    def transform(self, xx):
        rr = []
        for x in xx:
            r = self.func(x)
            rr.append(r)
        return rr

    def inverse_transform(self, xx):
        rr = []
        for x in xx:
            r = self.inverse(x)
            rr.append(r)
        return rr
