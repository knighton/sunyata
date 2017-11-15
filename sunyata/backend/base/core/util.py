from ..base import APIMixin


class BaseUtilAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def to_shape(self, x, ndim):
        if isinstance(x, int):
            assert 0 <= x
            assert 1 <= ndim
            x = (x,) * ndim
        elif isinstance(x, tuple):
            assert len(x) == ndim
        else:
            assert False
        return x

    def to_one(self, x):
        return self.to_shape(x, 1)[0]
