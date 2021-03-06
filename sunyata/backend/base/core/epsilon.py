from ..base import APIMixin


class BaseEpsilonAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)
        self.set_epsilon(1e-5)

    def set_epsilon(self, epsilon):
        assert isinstance(epsilon, float)
        assert 0 < epsilon < 1e-2
        self._epsilon = epsilon

    def epsilon(self):
        return self._epsilon
