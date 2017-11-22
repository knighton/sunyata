from ...base import APIMixin


class BaseGaussianNoiseAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def _gaussian_noise(self, x, is_training, std, keep_axis, ndim):
        if not is_training:
            return x
        x_shape = self.shape(x)
        if ndim is not None:
            assert len(x_shape) == ndim + 2
        mul_shape = self._dropout_mask_shape(x_shape, keep_axis)
        mul = self.random_normal(mul_shape, 0, std)
        return x + self.constant(mul)

    def gaussian_noise(self, x, is_training, std, keep_axis=None):
        return self._gaussian_noise(x, is_training, std, keep_axis, None)

    def gaussian_noise0d(self, x, is_training, std, keep_axis=None):
        return self._gaussian_noise(x, is_training, std, keep_axis, 0)

    def gaussian_noise1d(self, x, is_training, std, keep_axis=None):
        return self._gaussian_noise(x, is_training, std, keep_axis, 1)

    def gaussian_noise2d(self, x, is_training, std, keep_axis=None):
        return self._gaussian_noise(x, is_training, std, keep_axis, 2)

    def gaussian_noise3d(self, x, is_training, std, keep_axis=None):
        return self._gaussian_noise(x, is_training, std, keep_axis, 3)
