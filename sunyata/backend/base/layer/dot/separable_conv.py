from ...base import APIMixin


class BaseSeparableConvAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def separable_conv(self, x, depthwise_kernel, pointwise_kernel, bias,
                       stride, pad, dilation):
        raise NotImplementedError

    def separable_conv1d(self, x, depthwise_kernel, pointwise_kernel, bias,
                         stride, pad, dilation):
        raise NotImplementedError

    def separable_conv2d(self, x, depthwise_kernel, pointwise_kernel, bias,
                         stride, pad, dilation):
        raise NotImplementedError

    def separable_conv3d(self, x, depthwise_kernel, pointwise_kernel, bias,
                         stride, pad, dilation):
        raise NotImplementedError

    def separable_conv_out_shape(self, in_shape, out_channels, face, stride,
                                 pad, dilation):
        return self.conv_out_shape(
            in_shape, out_channels, face, stride, pad, dilation)
