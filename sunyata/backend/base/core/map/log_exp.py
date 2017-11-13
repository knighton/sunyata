import numpy as np

from ...base import APIMixin


_LOG_2 = np.log(2)
_LOG_10 = np.log(10)


class BaseLogExpAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def exp(self, x):
        raise NotImplementedError

    def expm1(self, x):
        return self.exp(x) - 1

    def log(self, x):
        raise NotImplementedError

    def log2(self, x):
        return self.log(x) / _LOG_2

    def log10(self, x):
        return self.log(x) / _LOG_10

    def log1p(self, x):
        return self.log(x + 1)
