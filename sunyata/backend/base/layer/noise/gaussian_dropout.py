from ...base import APIMixin


class BaseGaussianDropoutAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def _gaussian_dropout(self, x, train, rate, keep_axis, ndim):
        if not train:
            return x
        x_shape = self.shape(x)
        if ndim is not None:
            assert len(x_shape) == ndim + 2
        mul_shape = self._dropout_mask_shape(x_shape, keep_axis)
        std = (rate / (1 - rate)) ** 0.5
        mul = self.random_normal(mul_shape, 1, std)
        return x * self.constant(mul)

    def gaussian_dropout(self, x, train, rate=0.5, keep_axis=None):
        return self._gaussian_dropout(x, train, rate, keep_axis, None)

    def gaussian_dropout0d(self, x, train, rate=0.5, keep_axis=None):
        return self._gaussian_dropout(x, train, rate, keep_axis, 0)

    def gaussian_dropout1d(self, x, train, rate=0.5, keep_axis=None):
        return self._gaussian_dropout(x, train, rate, keep_axis, 1)

    def gaussian_dropout2d(self, x, train, rate=0.5, keep_axis=None):
        return self._gaussian_dropout(x, train, rate, keep_axis, 2)

    def gaussian_dropout3d(self, x, train, rate=0.5, keep_axis=None):
        return self._gaussian_dropout(x, train, rate, keep_axis, 3)
