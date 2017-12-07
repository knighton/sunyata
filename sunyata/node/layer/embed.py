from ... import backend as Z
from ... import init
from .base import Form, node_wrap, TransformLayer, TransformSpec


class EmbedLayer(TransformLayer):
    def __init__(self, reference, x_ndim):
        super().__init__(x_ndim)
        self.reference = self.add_param(reference)

    def transform(self, x, train):
        return Z.embed(x, self.reference)


class EmbedSpec(TransformSpec):
    def __init__(self, vocab, channels, dtype=None, reference_init='uniform',
                 spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.vocab_size = vocab
        self.channels = channels
        self.dtype = Z.dtype(dtype)
        self.reference_init = init.get(reference_init)

    def build_transform(self, form):
        reference_shape = self.vocab_size, self.channels
        reference = self.reference_init(reference_shape, self.dtype)
        layer = EmbedLayer(reference, self.x_ndim())
        in_len, = form.batch_shape
        out_shape = self.channels, in_len
        form = Form(out_shape, self.dtype)
        return layer, form


node_wrap(EmbedSpec, (None, 1, 2, 3))
