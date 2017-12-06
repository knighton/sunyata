from .... import backend as Z
from ..base import node_wrap, TransformLayer, TransformSpec


class BentIdentityLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.bent_identity(x)


class BentIdentitySpec(TransformSpec):
    def build_transform(self, form):
        return BentIdentityLayer(self.x_ndim()), form


node_wrap(BentIdentitySpec)
