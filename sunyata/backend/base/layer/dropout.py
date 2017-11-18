from ..base import APIMixin


class BaseDropoutAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def _dropout(self, x, is_training, rate, noise_shape):
        if is_training:
            mask = self.random_binomial(noise_shape, rate, self.dtype_of(x),
                                        self.device_of(x))
            x = x * mask / (1 - rate)
        return x

    def vanilla_dropout(self, x, is_training, rate):
        noise_shape = self.shape(x)
        return self._dropout(x, is_training, rate, noise_shape)

    def vanilla_dropout0d(self, x, is_training, rate):
        assert self.ndim(x) == 2
        return self.vanilla_dropout(x, is_training, rate)

    def vanilla_dropout1d(self, x, is_training, rate):
        assert self.ndim(x) == 3
        return self.vanilla_dropout(x, is_training, rate)

    def vanilla_dropout2d(self, x, is_training, rate):
        assert self.ndim(x) == 4
        return self.vanilla_dropout(x, is_training, rate)

    def vanilla_dropout3d(self, x, is_training, rate):
        assert self.ndim(x) == 5
        return self.vanilla_dropout(x, is_training, rate)
