from ..base import APIMixin


class BaseShapeAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def ndim(self, x):
        raise NotImplementedError

    def shape(self, x):
        raise NotImplementedError

    def size(self, x):
        raise NotImplementedError

    def reshape(self, x, shape):
        raise NotImplementedError

    def expand_dims(self, x, axis):
        raise NotImplementedError
