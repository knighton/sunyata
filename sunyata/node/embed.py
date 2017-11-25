import numpy as np

from .. import backend as Z
from .base import Form, TransformLayer, TransformSpec


class EmbedLayer(TransformLayer):
    def __init__(self, reference):
        super().__init__()
        self.reference = Z.variable(Z.numpy_to_device(reference))

    def params(self):
        return [self.reference]

    def forward_one(self, x, is_training):
        return Z.embed(x, self.reference)


class EmbedSpec(TransformSpec):
    def __init__(self, vocab, channels, dtype=None):
        super().__init__()
        self.vocab_size = vocab
        self.channels = channels
        self.dtype = Z.dtype(dtype)

    def build_one(self, form):
        reference_shape = self.vocab_size, self.channels
        reference = np.random.uniform(0, 1, reference_shape).astype(self.dtype)
        layer = EmbedLayer(reference)
        in_len, = form.shape
        out_shape = self.channels, in_len
        form = Form(out_shape, self.dtype)
        return layer, form
