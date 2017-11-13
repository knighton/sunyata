from chainer import functions as F

from ....base.core.map.log_exp import BaseLogExpAPI


class ChainerLogExpAPI(BaseLogExpAPI):
    def __init__(self):
        BaseLogExpAPI.__init__(self)

    def exp(self, x):
        return F.exp(x)

    def log(self, x):
        return F.log(x)

    def log2(self, x):
        return F.log2(x)

    def log10(self, x):
        return F.log10(x)

    def log1p(self, x):
        return F.log1p(x)
