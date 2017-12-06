from .... import backend as Z
from .... import init
from ..base import Form, TransformLayer, TransformSpec


class SeparableConvLayer(TransformLayer):
    def __init__(self, depthwise_kernel, pointwise_kernel, bias, stride, pad,
                 dilation, x_ndim=None):
        super().__init__(x_ndim)
        self.depthwise_kernel = self.add_param(depthwise_kernel)
        self.pointwise_kernel = self.add_param(pointwise_kernel)
        if bias is None:
            self.bias = None
        else:
            self.bias = self.add_param(bias)
        self.stride = stride
        self.pad = pad
        self.dilation = dilation

    def transform(self, x, is_training):
        return Z.separable_conv(x, self.depthwise_kernel, self.pointwise_kernel,
                                self.bias, self.stride, self.pad, self.dilation)


class SeparableConvSpec(TransformSpec):
    def __init__(self, channels=None, depth_mul=1, face=3, stride=1, pad='same',
                 dilation=1, has_bias=True,
                 depthwise_kernel_init='glorot_uniform',
                 pointwise_kernel_init='glorot_uniform', bias_init='zeros',
                 spatial_ndim=None):
        super().__init__(spatial_ndim)
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

    def build_transform(self, form):
        depthwise_in_channels = form.batch_shape[0]
        depthwise_out_channels = self.depth_mul
        pointwise_in_channels = depthwise_in_channels * depthwise_out_channels
        if self.channels is None:
            pointwise_out_channels = depthwise_in_channels
        else:
            pointwise_out_channels = self.channels
        face = Z.to_shape(self.face, self.spatial_ndim())
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
            self.dilation, self.x_ndim())
        out_shape = Z.separable_conv_out_shape(
            form.batch_shape, pointwise_out_channels, face, self.stride,
            self.pad, self.dilation)
        form = Form(out_shape, form.dtype)
        return layer, form
