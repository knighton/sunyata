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
        stride = self.to_shape(stride, ndim)
        dilation = self.to_shape(dilation, ndim)
        assert pad == 'SAME'
        x = self._to_channels_last(x)
        x = tf.nn.convolution(x, kernel, pad, stride, dilation)
        x = self._to_channels_first(x)
        broadcasted = (1,) + self.shape(bias) + (1,) * ndim
        return x + self.reshape(bias, broadcasted)

    def conv1d(self, x, kernel, bias, stride, pad, dilation):
        return self.conv(x, kernel, bias, stride, pad, dilation)

    def conv2d(self, x, kernel, bias, stride, pad, dilation):
        return self.conv(x, kernel, bias, stride, pad, dilation)

    def conv3d(self, x, kernel, bias, stride, pad, dilation):
        return self.conv(x, kernel, bias, stride, pad, dilation)
