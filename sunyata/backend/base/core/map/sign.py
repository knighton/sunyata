from ...base import APIMixin


class BaseSignAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def abs(self, x):
        return self.maximum(x, -1 * x)

    def neg(self, x):
        return -1 * x

    def sign(self, x):
        return self.less(0, x) * 2 - 1
