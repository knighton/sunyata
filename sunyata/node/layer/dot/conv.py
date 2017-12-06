from .... import backend as Z
from .... import init
from ..base import Form, node_wrap, TransformLayer, TransformSpec


class ConvLayer(TransformLayer):
    def __init__(self, kernel, bias, stride, pad, dilation, x_ndim):
        super().__init__(x_ndim)
        self.kernel = self.add_param(kernel)
        if bias is None:
            self.bias = None
        else:
            self.bias = self.add_param(bias)
        self.stride = stride
        self.pad = pad
        self.dilation = dilation

    def transform(self, x, is_training):
        return Z.conv(x, self.kernel, self.bias, self.stride, self.pad,
                      self.dilation)


class ConvSpec(TransformSpec):
    def __init__(self, channels=None, face=3, stride=1, pad='same', dilation=1,
                 has_bias=True, kernel_init='glorot_uniform', bias_init='zeros',
                 spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.channels = channels
        self.face = face
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.has_bias = has_bias
        self.kernel_init = init.get(kernel_init)
        self.bias_init = init.get(bias_init)

    def build_transform(self, form):
        in_channels = form.shape[0]
        out_channels = in_channels if self.channels is None else self.channels
        face = Z.to_shape(self.face, self.spatial_ndim())
        kernel_shape = (out_channels, in_channels) + face
        kernel = self.kernel_init(kernel_shape, form.dtype, 'conv_kernel')
        if self.has_bias:
            bias_shape = out_channels,
            bias = self.bias_init(bias_shape, form.dtype)
        else:
            bias = None
        layer = ConvLayer(
            kernel, bias, self.stride, self.pad, self.dilation, self.x_ndim())
        out_shape = Z.conv_out_shape(
            form.shape, out_channels, face, self.stride, self.pad,
            self.dilation)
        form = Form(out_shape, form.dtype)
        return layer, form


node_wrap(ConvSpec, (None, 1, 2, 3))
