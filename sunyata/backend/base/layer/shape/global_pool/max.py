from ....base import APIMixin


class BaseGlobalMaxPoolAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def _global_max_pool(self, x, ndim):
        if ndim is None:
            ndim + 2 == self.ndim(x)
        else:
            assert self.ndim(x) == ndim + 2
        axes = list(range(ndim))[2:]
        return self.max(x, axes)

    def global_max_pool(self, x):
        return self._global_max_pool(x, None)

    def global_max_pool1d(self, x):
        return self._global_max_pool(x, 1)

    def global_max_pool2d(self, x):
        return self._global_max_pool(x, 2)

    def global_max_pool3d(self, x):
        return self._global_max_pool(x, 3)
