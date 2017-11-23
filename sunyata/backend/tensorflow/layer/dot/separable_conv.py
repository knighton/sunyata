import tensorflow as tf

from ....base.layer.dot.separable_conv import BaseSeparableConvAPI


class TensorFlowSeparableConvAPI(BaseSeparableConvAPI):
    def __init__(self):
        BaseSeparableConvAPI.__init__(self)
        self._ndim2separable_conv = {
            1: self.separable_conv1d,
            2: self.separable_conv2d,
            3: self.separable_conv3d,
        }

    def separable_conv(self, x, depthwise_kernel, pointwise_kernel, bias,
                       stride, pad, dilation):
        ndim = self.ndim(x) - 2
        return self._ndim2separable_conv[ndim](
            x, depthwise_kernel, pointwise_kernel, bias, stride, pad, dilation)

    def separable_conv2d(self, x, depthwise_kernel, pointwise_kernel, bias,
                         stride, pad, dilation):
        assert self.ndim(x) == 4
        spatial_ndim = len(x.shape) - 2
        face = self.shape(depthwise_kernel)[:-2]
        stride = self.to_shape(stride, spatial_ndim)
        pre_pad, conv_word_pad = self.unpack_conv_pad_to_word(
            face, pad, dilation)
        if pre_pad is not None:
            x = self.constant_pad(x, pre_pad, 0)
        x = self._to_channels_last(x)
        x = tf.nn.separable_conv2d(
            x, depthwise_kernel, pointwise_kernel, stride, conv_word_pad,
            dilation)
        x = self._to_channels_first(x)
        if bias is not None:
            bias_shape = (1,) + self.shape(bias) + (1,) * spatial_ndim
            x += self.reshape(bias, bias_shape)
        return x
