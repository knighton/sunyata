from ...base import APIMixin


class BaseRoundAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def ceil(self, x):
        raise NotImplementedError

    def floor(self, x):
        raise NotImplementedError

    def round(self, x):
        return self.floor(x + 0.5)

    def trunc(self, x):
        is_pos = self.less(0, x)
        return self.where(is_pos, self.floor(x), -self.floor(-x))
