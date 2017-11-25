from ..base import APIMixin


class BaseLinearAlgebraAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def matmul(self, a, b):
        raise NotImplementedError
