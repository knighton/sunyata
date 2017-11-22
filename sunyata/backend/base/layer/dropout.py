from ..base import APIMixin


class BaseDropoutAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def _spatial_dropout_mask_shape(self, x_shape, keep_axis):
        if keep_axis is None:
            mask_shape = x_shape
        else:
            if isinstance(keep_axis, int):
                keep_axes = [keep_axis]
            elif isinstance(keep_axis, (list, tuple)):
                keep_axes = keep_axis
            else:
                assert False
            mask_shape = [1] * len(x_shape)
            mask_shape[0] = x_shape[0]
            for axis in keep_axes:
                mask_shape[axis] = x_shape[axis]
            mask_shape = tuple(mask_shape)
        return mask_shape

    def spatial_dropout(self, x, is_training, keep_axis, rate):
        if not is_training:
            return x
        mask_shape = self._spatial_dropout_mask_shape(self.shape(x), keep_axis)
        mask = self.random_binomial(
            mask_shape, rate, self.dtype_of(x), self.device_of(x))
        mask = self.constant(mask)
        return x * mask / (1 - rate)

    def spatial_dropout0d(self, x, is_training, keep_axis, rate):
        assert self.ndim(x) == 2
        return self.spatial_dropout(x, is_training, keep_axis, rate)

    def spatial_dropout1d(self, x, is_training, keep_axis, rate):
        assert self.ndim(x) == 3
        return self.spatial_dropout(x, is_training, keep_axis, rate)

    def spatial_dropout2d(self, x, is_training, keep_axis, rate):
        assert self.ndim(x) == 4
        return self.spatial_dropout(x, is_training, keep_axis, rate)

    def spatial_dropout3d(self, x, is_training, keep_axis, rate):
        assert self.ndim(x) == 5
        return self.spatial_dropout(x, is_training, keep_axis, rate)

    def vanilla_dropout(self, x, is_training, rate):
        return self.spatial_dropout(x, is_training, None, rate)

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
