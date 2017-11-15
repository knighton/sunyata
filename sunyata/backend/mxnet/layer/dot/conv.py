import mxnet as mx

from ....base.layer.dot.conv import BaseConvAPI


class MXNetConvAPI(BaseConvAPI):
    def __init__(self):
        BaseConvAPI.__init__(self)

    def conv(self, x, kernel, bias, stride, pad, dilation):
        ndim = x.ndim - 2
        stride = self.to_shape(stride, ndim)
        dilation = self.to_shape(dilation, ndim)
        pad = self.unpack_conv_pad(kernel, pad, dilation)
        pre_pad, conv_single_pad = self.conv_pad_to_singles(pad)
        if pre_pad is not None:
            x = self.constant_pad(x, pre_pad, 0)
        return mx.nd.Convolution(
            x, kernel, bias, kernel.shape[2:], stride, dilation,
            conv_single_pad, kernel.shape[0])

    def conv1d(self, x, kernel, bias, stride, pad, dilation):
        return self.conv(x, kernel, bias, stride, pad, dilation)

    def conv2d(self, x, kernel, bias, stride, pad, dilation):
        return self.conv(x, kernel, bias, stride, pad, dilation)

    def conv3d(self, x, kernel, bias, stride, pad, dilation):
        return self.conv(x, kernel, bias, stride, pad, dilation)
