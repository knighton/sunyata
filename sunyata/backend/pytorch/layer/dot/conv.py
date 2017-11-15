from torch.nn import functional as F

from ....base.layer.dot.conv import BaseConvAPI


class PyTorchConvAPI(BaseConvAPI):
    def __init__(self):
        BaseConvAPI.__init__(self)
        self._ndim2conv = {
            1: self.conv1d,
            2: self.conv2d,
            3: self.conv3d,
        }

    def conv(self, x, kernel, bias, stride, pad, dilation):
        ndim = x.dim() - 2
        return self._ndim2conv[ndim](x, kernel, bias, stride, pad, dilation)

    def _conv(self, func_name, x, kernel, bias, stride, pad, dilation):
        ndim = self.ndim(x) - 2
        face = self.shape(kernel)[2:]
        stride = self.to_shape(stride, ndim)
        dilation = self.to_shape(dilation, ndim)
        pre_pad, conv_singles_pad = \
            self.unpack_conv_pad_to_singles(face, pad, dilation)
        func = getattr(F, func_name)
        if ndim == 1:
            stride, = stride
            conv_singles_pad, = conv_singles_pad
            dilation, = dilation
        if pre_pad is not None:
            x = self.constant_pad(x, pre_pad, 0)
        return func(x, kernel, bias, stride, conv_singles_pad, dilation)

    def conv1d(self, x, kernel, bias, stride, pad, dilation):
        return self._conv('conv1d', x, kernel, bias, stride, pad, dilation)

    def conv2d(self, x, kernel, bias, stride, pad, dilation):
        return self._conv('conv2d', x, kernel, bias, stride, pad, dilation)

    def conv3d(self, x, kernel, bias, stride, pad, dilation):
        return self._conv('conv3d', x, kernel, bias, stride, pad, dilation)
