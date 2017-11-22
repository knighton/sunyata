import numpy as np

from ...base import APIMixin


class BaseConvAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def conv(self, x, kernel, bias, stride, pad, dilation):
        raise NotImplementedError

    def conv1d(self, x, kernel, bias, stride, pad, dilation):
        raise NotImplementedError

    def conv2d(self, x, kernel, bias, stride, pad, dilation):
        raise NotImplementedError

    def conv3d(self, x, kernel, bias, stride, pad, dilation):
        raise NotImplementedError

    def conv_out_shape(self, in_shape, out_channels, face, stride, pad,
                       dilation):
        spatial_ndim = len(in_shape) - 1
        face = self.to_shape(face, spatial_ndim)
        stride = self.to_shape(stride, spatial_ndim)
        pad = self._unpack_conv_pad(face, pad, dilation)
        dilation = self.to_shape(dilation, spatial_ndim)
        out_shape = [out_channels]
        for i in range(len(in_shape) - 1):
            numerator = in_shape[i + 1] + pad[i][0] + pad[i][1] - \
                dilation[i] * (face[i] - 1) - 1
            dim = int(np.floor(numerator / stride[i])) + 1
            out_shape.append(dim)
        return tuple(out_shape)
