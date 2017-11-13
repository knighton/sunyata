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

    def squeeze(self, x, axis=None):
        raise NotImplementedError

    def tile(self, x, reps):
        raise NotImplementedError

    def transpose(self, x, axes):
        raise NotImplementedError

    def concat(self, xx, axis):
        raise NotImplementedError

    def stack(self, xx, axis=0):
        raise NotImplementedError
