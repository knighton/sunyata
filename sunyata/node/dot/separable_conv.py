from ... import backend as Z
from ... import init
from ..base import Form, TransformLayer, TransformSpec


class SeparableConvLayer(TransformLayer):
    def __init__(self, depthwise_kernel, pointwise_kernel, bias, stride, pad,
                 dilation, ndim):
        super().__init__(ndim)
        self.depthwise_kernel = Z.variable(Z.numpy_to_device(depthwise_kernel))
        self.pointwise_kernel = Z.variable(Z.numpy_to_device(pointwise_kernel))
        if bias is None:
            self.bias = None
        else:
            self.bias = Z.variable(Z.numpy_to_device(bias))
        self.stride = stride
        self.pad = pad
        self.dilation = dilation

    def params(self):
        params = [self.depthwise_kernel, self.pointwise_kernel]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def forward_one(self, x, is_training):
        return Z.separable_conv(x, self.depthwise_kernel, self.pointwise_kernel,
                                self.bias, self.stride, self.pad, self.dilation)


class SeparableConvSpec(TransformSpec):
    def __init__(self, channels=None, depth_mul=1, face=3, stride=1, pad='same',
                 dilation=1, has_bias=True,
                 depthwise_kernel_init='glorot_uniform',
                 pointwise_kernel_init='glorot_uniform', bias_init='zeros',
                 ndim=None):
        super().__init__(ndim)
        self.channels = channels
        self.depth_mul = depth_mul
        self.face = face
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.has_bias = has_bias
        self.depthwise_kernel_init = init.get(depthwise_kernel_init)
        self.pointwise_kernel_init = init.get(pointwise_kernel_init)
        self.bias_init = init.get(bias_init)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        depthwise_in_channels = form.shape[0]
        depthwise_out_channels = self.depth_mul
        pointwise_in_channels = depthwise_in_channels * depthwise_out_channels
        if self.channels is None:
            pointwise_out_channels = depthwise_in_channels
        else:
            pointwise_out_channels = self.channels
        face = Z.to_shape(self.face, ndim - 2)
        depthwise_kernel_shape = \
            (depthwise_out_channels, depthwise_in_channels) + face
        depthwise_kernel = self.depthwise_kernel_init(
            depthwise_kernel_shape, form.dtype, 'conv_kernel')
        pointwise_kernel_shape = \
            (pointwise_out_channels, pointwise_in_channels, 1, 1)
        pointwise_kernel = self.pointwise_kernel_init(
            pointwise_kernel_shape, form.dtype, 'conv_kernel')
        if self.has_bias:
            bias_shape = pointwise_out_channels,
            bias = self.bias_init(bias_shape, form.dtype)
        else:
            bias = None
        layer = SeparableConvLayer(
            depthwise_kernel, pointwise_kernel, bias, self.stride, self.pad,
            self.dilation, ndim)
        out_shape = Z.separable_conv_out_shape(
            form.shape, pointwise_out_channels, face, self.stride, self.pad,
            self.dilation)
        form = Form(out_shape, form.dtype)
        return layer, form
