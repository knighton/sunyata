from .... import backend as Z
from .... import init
from ..base import Form, LinkBuilder, TransformLayer, TransformSpec


class DenseLayer(TransformLayer):
    def __init__(self, kernel, bias):
        super().__init__(2)
        self.kernel = self.add_param(kernel)
        if bias is None:
            self.bias = None
        else:
            self.bias = self.add_param(bias)

    def transform(self, x, is_training):
        return Z.dense(x, self.kernel, self.bias)


class DenseSpec(TransformSpec):
    def __init__(self, dim=None, has_bias=True, kernel_init='glorot_uniform',
                 bias_init='zeros'):
        super().__init__(0)
        self.out_dim = dim
        self.has_bias = has_bias
        self.kernel_init = init.get(kernel_init)
        self.bias_init = init.get(bias_init)

    def build_transform(self, form):
        assert len(form.batch_shape) == 1
        in_dim, = form.batch_shape
        out_dim = in_dim if self.out_dim is None else self.out_dim
        kernel_shape = out_dim, in_dim
        kernel = self.kernel_init(kernel_shape, form.dtype, 'conv_kernel')
        if self.has_bias:
            bias_shape = out_dim,
            bias = self.bias_init(bias_shape, form.dtype)
        else:
            bias = None
        out_shape = out_dim,
        layer = DenseLayer(kernel, bias)
        form = Form(out_shape, form.dtype)
        return layer, form


Dense = LinkBuilder(DenseSpec)
