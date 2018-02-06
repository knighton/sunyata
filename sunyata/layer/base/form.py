from ... import backend as Z


class Form(object):
    def __init__(self, shape, dtype):
        assert isinstance(shape, tuple)
        for dim in shape:
            assert isinstance(dim, int)
            assert 1 <= dim
        assert dtype in Z.supported_dtypes()
        self.batch_shape = shape
        self.dtype = dtype

    @property
    def batch_ndim(self):
        return len(self.batch_shape)

    def check(self, x):
        assert Z.shape(x)[1:] == self.batch_shape
        assert Z.dtype_of(x) == self.dtype
