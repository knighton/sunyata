from ... import backend as Z


class Form(object):
    def __init__(self, shape, dtype):
        assert isinstance(shape, tuple)
        for dim in shape:
            assert isinstance(dim, int)
            assert 1 <= dim
        assert dtype in Z.supported_dtypes()
        self.shape = shape
        self.dtype = dtype

    def check(self, x):
        assert Z.shape(x)[1:] == self.shape
        assert Z.dtype_of(x) == self.dtype


class Spec(object):
    def build_multi(self, forms):
        raise NotImplementedError


class MergeSpec(Spec):
    pass


class TransformSpec(Spec):
    def __init__(self, ndim):
        super().__init__()
        assert ndim in {None, 0, 1, 2, 3}
        self.in_spatial_ndim = ndim

    def in_ndim(self, batch_shape):
        ndim = len(batch_shape) + 1
        if self.in_spatial_ndim is not None:
            assert ndim == self.in_spatial_ndim + 2
        return ndim

    def build_one(self, form):
        raise NotImplementedError

    def build_multi(self, forms):
        assert len(forms) == 1
        form, = forms
        layer, form = self.build_one(form)
        return layer, [form]
