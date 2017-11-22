from ..base import APIMixin


class BaseDropoutAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def spatial_dropout(self, x, is_training, mask_shape, rate):
        if not is_training:
            return x
        mask = self.random_binomial(
            mask_shape, rate, self.dtype_of(x), self.device_of(x))
        return x * mask / (1 - rate)

    def spatial_dropout0d(self, x, is_training, mask_shape, rate):
        assert self.ndim(x) == 2
        return self.spatial_dropout(x, is_training, mask_shape, rate)

    def spatial_dropout1d(self, x, is_training, mask_shape, rate):
        assert self.ndim(x) == 3
        return self.spatial_dropout(x, is_training, mask_shape, rate)

    def spatial_dropout2d(self, x, is_training, mask_shape, rate):
        assert self.ndim(x) == 4
        return self.spatial_dropout(x, is_training, mask_shape, rate)

    def spatial_dropout3d(self, x, is_training, mask_shape, rate):
        assert self.ndim(x) == 5
        return self.spatial_dropout(x, is_training, mask_shape, rate)

    def vanilla_dropout(self, x, is_training, rate):
        mask_shape = self.shape(x)
        return self.spatial_dropout(x, is_training, mask_shape, rate)

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
