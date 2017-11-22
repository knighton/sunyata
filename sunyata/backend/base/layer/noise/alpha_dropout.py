from ...base import APIMixin


class BaseAlphaDropoutAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        self._alpha_p = -alpha * scale

    def _alpha_dropout(self, x, is_training, rate, keep_axis, ndim):
        if not is_training:
            return x
        x_shape = self.shape(x)
        if ndim is not None:
            assert len(x_shape) == ndim + 2
        mask_shape = self._dropout_mask_shape(x_shape, keep_axis)
        mask = self.random_binomial(
            mask_shape, rate, self.dtype_of(x), self.device_of(x))
        mask = self.constant(mask)
        a = ((1 - rate) * (1 + rate * self._alpha_p ** 2)) ** -0.5
        b = -a * self._alpha_p * rate
        x = mask * x + (1 - mask) * self._alpha_p
        return a * x + b

    def alpha_dropout(self, x, is_training, rate=0.5, keep_axis=None):
        return self._alpha_dropout(x, is_training, rate, keep_axis, None)

    def alpha_dropout0d(self, x, is_training, rate=0.5, keep_axis=None):
        return self._alpha_dropout(x, is_training, rate, keep_axis, 0)

    def alpha_dropout1d(self, x, is_training, rate=0.5, keep_axis=None):
        return self._alpha_dropout(x, is_training, rate, keep_axis, 1)

    def alpha_dropout2d(self, x, is_training, rate=0.5, keep_axis=None):
        return self._alpha_dropout(x, is_training, rate, keep_axis, 2)

    def alpha_dropout3d(self, x, is_training, rate=0.5, keep_axis=None):
        return self._alpha_dropout(x, is_training, rate, keep_axis, 3)
