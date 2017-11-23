from chainer import functions as F

from ....base.layer.dot.depthwise_conv import BaseDepthwiseConvAPI


class ChainerDepthwiseConvAPI(BaseDepthwiseConvAPI):
    def __init__(self):
        BaseDepthwiseConvAPI.__init__(self)
        self._ndim2depthwise_conv = {
            1: self.depthwise_conv1d,
            2: self.depthwise_conv2d,
            3: self.depthwise_conv3d,
        }

    def depthwise_conv(self, x, kernel, bias, stride, pad, dilation):
        ndim = self.ndim(x) - 2
        return self._ndim2depthwise_conv[ndim](
            x, kernel, bias, stride, pad, dilation)

    def depthwise_conv2d(self, x, kernel, bias, stride, pad, dilation):
        assert self.ndim(x) == 4
        spatial_ndim = len(x.shape) - 2
        face = self.shape(kernel)[2:]
        stride = self.to_shape(stride, spatial_ndim)
        pre_pad, conv_singles_pad = \
            self.unpack_conv_pad_to_singles(face, pad, dilation)
        if pre_pad is not None:
            x = self.constant_pad(x, pre_pad, 0)
        dilation = self.to_shape(dilation, spatial_ndim)
        for dim in dilation:
            assert dim == 1
        return F.depthwise_conv_2d(x, kernel, bias, stride, conv_singles_pad)
