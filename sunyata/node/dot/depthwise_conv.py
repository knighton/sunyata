import numpy as np

from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class DepthwiseConvLayer(TransformLayer):
    def __init__(self, depthwise_kernel, bias, stride, pad, dilation, ndim):
        super().__init__(ndim)
        self.depthwise_kernel = Z.variable(Z.numpy_to_device(depthwise_kernel))
        if bias is None:
            self.bias = None
        else:
            self.bias = Z.variable(Z.numpy_to_device(bias))
        self.stride = stride
        self.pad = pad
        self.dilation = dilation

    def params(self):
        params = [self.depthwise_kernel]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def forward_one(self, x, is_training):
        return Z.depthwise_conv(x, self.depthwise_kernel, self.bias,
                                self.stride, self.pad, self.dilation)


class DepthwiseConvSpec(TransformSpec):
    def __init__(self, depth_mul=1, face=3, stride=1, pad='same', dilation=1,
                 has_bias=True, ndim=None):
        super().__init__(ndim)
        self.depth_mul = depth_mul
        self.face = face
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.has_bias = has_bias

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        depthwise_in_channels = form.shape[0]
        depthwise_out_channels = self.depth_mul
        face = Z.to_shape(self.face, ndim - 2)
        depthwise_kernel_shape = \
            (depthwise_out_channels, depthwise_in_channels) + face
        depthwise_kernel = \
            np.random.normal(0, 0.1, depthwise_kernel_shape).astype(form.dtype)
        if self.has_bias:
            bias_shape = depthwise_in_channels * depthwise_out_channels,
            bias = np.zeros(bias_shape, form.dtype)
        else:
            bias = None
        layer = DepthwiseConvLayer(
            depthwise_kernel, bias, self.stride, self.pad, self.dilation, ndim)
        out_shape = Z.depthwise_conv_out_shape(
            form.shape, face, self.stride, self.pad, self.dilation)
        form = Form(out_shape, form.dtype)
        return layer, form
