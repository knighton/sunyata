import numpy as np

from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class ConvLayer(TransformLayer):
    def __init__(self, kernel, bias, stride, pad, dilation):
        TransformLayer.__init__(self)
        self.kernel = Z.variable(Z.numpy_to_device(kernel))
        self.bias = Z.variable(Z.numpy_to_device(bias))
        self.stride = stride
        self.pad = pad
        self.dilation = dilation

    def params(self):
        return [self.kernel, self.bias]

    def forward(self, x):
        return Z.conv(x, self.kernel, self.bias, self.stride, self.pad,
                      self.dilation)


class ConvSpec(TransformSpec):
    def __init__(self, channels=None, face=3, stride=1, pad='same', dilation=1,
                 ndim=None):
        super().__init__(ndim)
        self.channels = channels
        self.face = face
        self.stride = stride
        self.pad = pad
        self.dilation = dilation

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        out_channels = self.channels
        in_channels = form.shape[0]
        face = Z.to_shape(self.face, ndim - 2)
        kernel_shape = (out_channels, in_channels) + face
        kernel = np.random.normal(0, 0.1, kernel_shape).astype(form.dtype)
        bias_shape = out_channels,
        bias = np.random.normal(0, 0.1, bias_shape).astype(form.dtype)
        layer = ConvLayer(
            kernel, bias, self.stride, self.pad, self.dilation, ndim)
        form = Z.conv_out_shape(
            form.shape, out_channels, face, self.stride, self.pad,
            self.dilation)
        return layer, form
