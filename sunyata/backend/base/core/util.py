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

    def _unpack_conv_pad(self, face, pad, dilation):
        ndim = len(face)
        if pad == 'same':
            dilation = self.to_shape(dilation, ndim)
            pad = []
            for face_dim, dilation_dim in zip(face, dilation):
                left = (face_dim - 1) // 2 * dilation_dim
                right = face_dim // 2 * dilation_dim
                pad.append((left, right))
            pad = tuple(pad)
        elif pad == 'valid':
            pad = ((0, 0),) * ndim
        else:
            pad = self.unpack_int_pad(pad, ndim)
        return pad

    def _unpacked_conv_pad_to_singles(self, unpacked_pad):
        has_pre_pad = False
        pre_pad = []
        conv_singles_pad = []
        for left, right in unpacked_pad:
            dim = min(left, right)
            conv_singles_pad.append(dim)
            left -= dim
            right -= dim
            if left or right:
                has_pre_pad = True
            pre_pad.append((left, right))
        if has_pre_pad:
            pre_pad = tuple(pre_pad)
        else:
            pre_pad = None
        conv_singles_pad = tuple(conv_singles_pad)
        return pre_pad, conv_singles_pad

    def _unpacked_conv_pad_to_word(self, face, dilation, unpacked_pad):
        ndim = len(face)
        dilation = self.to_shape(dilation, ndim)
        could_be_same = True
        could_be_valid = True
        for (pad_left, pad_right), face_dim, dilation_dim in \
                zip(unpacked_pad, face, dilation):
            if (face_dim - 1) // 2 * dilation != pad_left:
                could_be_same = False
            if pad_left:
                could_be_valid = False
            if face_dim // 2 * dilation != pad_right:
                could_be_same = False
            if pad_right:
                could_be_valid = False
            if not could_be_same and not could_be_valid:
                break
        if could_be_same:
            pre_pad = None
            conv_word_pad = 'SAME'
        elif could_be_valid:
            pre_pad = None
            conv_word_pad = 'VALID'
        else:
            pre_pad = unpacked_pad
            conv_word_pad = 'VALID'
        return pre_pad, conv_word_pad

    def unpack_conv_pad_to_singles(self, face, pad, dilation):
        unpacked_pad = self._unpack_conv_pad(face, pad, dilation)
        return self._unpacked_conv_pad_to_singles(unpacked_pad)

    def unpack_conv_pad_to_word(self, face, pad, dilation):
        unpacked_pad = self._unpack_conv_pad(face, pad, dilation)
        return self._unpacked_conv_pad_to_word(face, dilation, unpacked_pad)
