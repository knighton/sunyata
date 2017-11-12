from ...base.core.reduce import BaseReduceAPI


class PyTorchReduceAPI(BaseReduceAPI):
    def argmin(self, x, axis=-1):
        return x.min(axis)[1]

    def argmax(self, x, axis=-1):
        return x.max(axis)[1]

    def _normalize_axis(self, axis, keepdims, ndim):
        if axis is None:
            if keepdims:
                axes = list(range(ndim))
            else:
                return None
        elif isinstance(axis, int):
            axes = [axis]
        elif isinstance(axis, tuple):
            axes = list(axis)
        elif isinstance(axis, list):
            pass
        else:
            assert False
        axes = list(map(lambda n: n % ndim, axes))
        return sorted(axes)

    def _reduce(self, name, x, axis=None, keepdims=False):
        axes = self._normalize_axis(axis, keepdims, x.dim())
        if axes is None:
            return getattr(x, name)(None, True)
        for axis in reversed(axes):
            if x.dim() == 1:
                keepdims = True
            x = getattr(x, name)(axis, keepdims)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def min(self, x, axis=None, keepdims=False):
        return self._reduce('min', x, axis, keepdims)

    def max(self, x, axis=None, keepdims=False):
        return self._reduce('max', x, axis, keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return self._reduce('mean', x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)

    def prod(self, x, axis=None, keepdims=False):
        return self._reduce('prod', x, axis, keepdims)
