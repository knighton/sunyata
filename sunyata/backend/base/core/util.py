from ..base import APIMixin


class BaseUtilAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def to_shape(self, x, ndim):
        if isinstance(x, int):
            assert 0 <= x
            assert 1 <= ndim
            x = (x,) * ndim
        elif isinstance(x, tuple):
            assert len(x) == ndim
        else:
            assert False
        return x

    def to_one(self, x):
        return self.to_shape(x, 1)[0]

    def unpack_pad(self, pad, ndim):
        if isinstance(pad, int):
            pad = ((pad, pad),) * ndim
        elif isinstance(pad, (list, tuple)):
            pad = list(pad)
            for i, x in enumerate(pad):
                if isinstance(x, int):
                    pad[i] = x, x
                elif isinstance(x, (list, tuple)):
                    assert len(x) == 2
                    assert isinstance(x[0], int)
                    assert isinstance(x[1], int)
                    pad[i] = tuple(x)
                else:
                    assert False
            pad = tuple(pad)
        else:
            assert False
        return pad
