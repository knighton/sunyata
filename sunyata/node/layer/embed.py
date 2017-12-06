from .. import backend as Z
from .. import init
from .base import Form, TransformLayer, TransformSpec


class EmbedLayer(TransformLayer):
    def __init__(self, reference):
        super().__init__()
        self.reference = self.add_param(reference)

    def forward_one(self, x, is_training):
        return Z.embed(x, self.reference)


class EmbedSpec(TransformSpec):
    def __init__(self, vocab, channels, dtype=None, reference_init='uniform'):
        super().__init__()
        self.vocab_size = vocab
        self.channels = channels
        self.dtype = Z.dtype(dtype)
        self.reference_init = init.get(reference_init)

    def build_one(self, form):
        reference_shape = self.vocab_size, self.channels
        reference = self.reference_init(reference_shape, self.dtype)
        layer = EmbedLayer(reference)
        in_len, = form.shape
        out_shape = self.channels, in_len
        form = Form(out_shape, self.dtype)
        return layer, form
