from ...base import APIMixin


class BaseDropoutAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def _dropout_mask_shape(self, x_shape, keep_axis):
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

    def _dropout(self, x, train, rate, keep_axis, ndim):
        if not train:
            return x
        x_shape = self.shape(x)
        if ndim is not None:
            assert len(x_shape) == ndim + 2
        mask_shape = self._dropout_mask_shape(x_shape, keep_axis)
        mask = self.random_binomial(
            mask_shape, rate, self.dtype_of(x), self.device_of(x))
        mask = self.constant(mask)
        return x * mask / (1 - rate)

    def dropout(self, x, train, rate=0.5, keep_axis=None):
        return self._dropout(x, train, rate, keep_axis, None)

    def dropout0d(self, x, train, rate=0.5, keep_axis=None):
        return self._dropout(x, train, rate, keep_axis, 0)

    def dropout1d(self, x, train, rate=0.5, keep_axis=None):
        return self._dropout(x, train, rate, keep_axis, 1)

    def dropout2d(self, x, train, rate=0.5, keep_axis=None):
        return self._dropout(x, train, rate, keep_axis, 2)

    def dropout3d(self, x, train, rate=0.5, keep_axis=None):
        return self._dropout(x, train, rate, keep_axis, 3)
