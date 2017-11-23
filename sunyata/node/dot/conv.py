import numpy as np

from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class ConvLayer(TransformLayer):
    def __init__(self, kernel, bias, stride, pad, dilation, ndim):
        super().__init__(ndim)
        self.kernel = Z.variable(Z.numpy_to_device(kernel))
        if bias is None:
            self.bias = None
        else:
            self.bias = Z.variable(Z.numpy_to_device(bias))
        self.stride = stride
        self.pad = pad
        self.dilation = dilation

    def params(self):
        if self.bias is None:
            variables = [self.kernel]
        else:
            variables = [self.kernel, self.bias]
        return variables

    def forward_one(self, x, is_training):
        return Z.conv(x, self.kernel, self.bias, self.stride, self.pad,
                      self.dilation)


class ConvSpec(TransformSpec):
    def __init__(self, channels=None, face=3, stride=1, pad='same', dilation=1,
                 has_bias=True, ndim=None):
        super().__init__(ndim)
        self.channels = channels
        self.face = face
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.has_bias = has_bias

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        in_channels = form.shape[0]
        out_channels = self.channels
        face = Z.to_shape(self.face, ndim - 2)
        kernel_shape = (out_channels, in_channels) + face
        kernel = np.random.normal(0, 0.1, kernel_shape).astype(form.dtype)
        if self.has_bias:
            bias_shape = out_channels,
            bias = np.zeros(bias_shape, form.dtype)
        else:
            bias = None
        layer = ConvLayer(
            kernel, bias, self.stride, self.pad, self.dilation, ndim)
        out_shape = Z.conv_out_shape(
            form.shape, out_channels, face, self.stride, self.pad,
            self.dilation)
        form = Form(out_shape, form.dtype)
        return layer, form
