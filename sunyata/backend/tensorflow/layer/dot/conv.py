import tensorflow as tf

from ....base.layer.dot.conv import BaseConvAPI


class TensorFlowConvAPI(BaseConvAPI):
    def __init__(self):
        BaseConvAPI.__init__(self)

    def _to_channels_last(self, x):
        ndim = len(x.shape)
        axes = [0] + list(range(2, ndim)) + [1]
        return tf.transpose(x, axes)

    def _to_channels_first(self, x):
        ndim = len(x.shape)
        axes = [0, ndim - 1] + list(range(1, ndim - 1))
        return tf.transpose(x, axes)

    def conv(self, x, kernel, bias, stride, pad, dilation):
        ndim = len(x.shape) - 2
        face = self.shape(kernel)[:-2]
        stride = self.to_shape(stride, ndim)
        dilation = self.to_shape(dilation, ndim)
        pre_pad, conv_word_pad = self.unpack_conv_pad_to_word(
            face, pad, dilation)
        if pre_pad is not None:
            x = self.constant_pad(x, pre_pad, 0)
        x = self._to_channels_last(x)
        x = tf.nn.convolution(x, kernel, conv_word_pad, stride, dilation)
        x = self._to_channels_first(x)
        broadcasted = (1,) + self.shape(bias) + (1,) * ndim
        return x + self.reshape(bias, broadcasted)

    def conv1d(self, x, kernel, bias, stride, pad, dilation):
        return self.conv(x, kernel, bias, stride, pad, dilation)

    def conv2d(self, x, kernel, bias, stride, pad, dilation):
        return self.conv(x, kernel, bias, stride, pad, dilation)

    def conv3d(self, x, kernel, bias, stride, pad, dilation):
        return self.conv(x, kernel, bias, stride, pad, dilation)
