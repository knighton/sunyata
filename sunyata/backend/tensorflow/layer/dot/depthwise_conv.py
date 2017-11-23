import tensorflow as tf

from ....base.layer.dot.conv import BaseDepthwiseConvAPI


class TensorFlowDepthwiseConvAPI(BaseDepthwiseConvAPI):
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
        face = self.shape(kernel)[:-2]
        stride = self.to_shape(stride, spatial_ndim)
        pre_pad, conv_word_pad = self.unpack_conv_pad_to_word(
            face, pad, dilation)
        if pre_pad is not None:
            x = self.constant_pad(x, pre_pad, 0)
        x = self._to_channels_last(x)
        x = tf.nn.depthwise_conv2d(x, kernel, stride, pad, dilation)
        x = self._to_channels_first(x)
        if bias is not None:
            bias_shape = (1,) + self.shape(bias) + (1,) * spatial_ndim
            x += self.reshape(bias, bias_shape)
        return x
