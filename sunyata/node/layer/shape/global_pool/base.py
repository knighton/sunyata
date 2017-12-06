from ..... import backend as Z
from ...base import Form, TransformLayer, TransformSpec


class GlobalPoolLayer(TransformLayer):
    pass


class GlobalPoolSpec(TransformSpec):
    def make_layer(self, form):
        raise NotImplementedError

    def build_transform(self, form):
        layer = self.make_layer(form)
        out_shape = Z.global_pool_out_shape(form.shape)
        form = Form(out_shape, form.dtype)
        return layer, form
