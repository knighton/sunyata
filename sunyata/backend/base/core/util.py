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

    def unpack_int_pad(self, pad, ndim):
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

    def unpack_conv_pad(self, kernel, pad, dilation):
        if pad == 'same':
            kernel_shape = self.shape(kernel)
            ndim = len(kernel_shape) - 2
            face = kernel_shape[2:]
            dilation = self.to_shape(dilation, ndim)
            pad = []
            for dim in face:
                left_right = (dim - 1) // 2, dim // 2
                pad.append(left_right)
            pad = tuple(pad)
        elif pad == 'valid':
            ndim = len(self.shape(kernel)) - 2
            pad = ((0, 0),) * ndim
        else:
            ndim = len(self.shape(kernel)) - 2
            pad = self.unpack_int_pad(pad, ndim)
        return pad

    def conv_pad_to_singles(self, pad):
        has_pre_pad = False
        pre_pad = []
        conv_single_pad = []
        for left, right in pad:
            dim = min(left, right)
            conv_single_pad.append(dim)
            left -= dim
            right -= dim
            if left or right:
                has_pre_pad = True
            pre_pad.append((left, right))
        if has_pre_pad:
            pre_pad = tuple(pre_pad)
        else:
            pre_pad = None
        conv_single_pad = tuple(conv_single_pad)
        return pre_pad, conv_single_pad
