import numpy as np

from .. import backend as Z
from .base import TransformLayer, TransformSpec


class ArcTanLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.arctan(x)


class ArcTanSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return ArcTanLayer(ndim), form


class BentIdentityLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.bent_identity(x)


class BentIdentitySpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return BentIdentityLayer(ndim), form


class ELULayer(TransformLayer):
    def __init__(self, alpha, ndim):
        super().__init__(ndim)
        self.alpha = alpha

    def forward_one(self, x, is_training):
        return Z.elu(x, self.alpha)


class ELUSpec(TransformSpec):
    def __init__(self, alpha=1., ndim=None):
        super().__init__(ndim)
        self.alpha = alpha

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return ELULayer(self.alpha, ndim), form


class HardShrinkLayer(TransformLayer):
    def __init__(self, lam, ndim):
        super().__init__(ndim)
        self.lam = lam

    def forward_one(self, x, is_training):
        return Z.hard_shrink(x, self.lam)


class HardShrinkSpec(TransformSpec):
    def __init__(self, lam=0.5, ndim=None):
        super().__init__(ndim)
        self.lam = lam

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return HardShrinkLayer(self.lam, ndim), form


class HardSigmoidLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.hard_sigmoid(x)


class HardSigmoidSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return HardSigmoidLayer(ndim), form


class HardTanhLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.hard_tanh(x)


class HardTanhSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return HardTanhLayer(ndim), form


class IdentityLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.identity(x)


class IdentitySpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return IdentityLayer(ndim), form


class LeakyReLULayer(TransformLayer):
    def __init__(self, alpha, ndim):
        super().__init__(ndim)
        self.alpha = alpha

    def forward_one(self, x, is_training):
        return Z.leaky_relu(x, self.alpha)


class LeakyReLUSpec(TransformSpec):
    def __init__(self, alpha=0.1, ndim=None):
        super().__init__(ndim)
        self.alpha = alpha

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return LeakyReLULayer(self.alpha, ndim), form


class LogSigmoidLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.log_sigmoid(x)


class LogSigmoidSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return LogSigmoidLayer(ndim), form


class ReLULayer(TransformLayer):
    def __init__(self, min, max, ndim):
        super().__init__(ndim)
        self.min = min
        self.max = max

    def forward_one(self, x, is_training):
        return Z.relu(x, self.min, self.max)


class ReLUSpec(TransformSpec):
    def __init__(self, min=0., max=np.inf, ndim=None):
        super().__init__(ndim)
        self.min = min
        self.max = max

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return ReLULayer(self.min, self.max, ndim), form


class SELULayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.selu(x)


class SELUSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return SELULayer(ndim), form


class SigmoidLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.sigmoid(x)


class SigmoidSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return SigmoidLayer(ndim), form


class SoftExponentialLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softexponential(x)


class SoftExponentialSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return SoftExponentialLayer(ndim), form


class SoftmaxLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softmax(x)


class SoftmaxSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return SoftmaxLayer(ndim), form


class SoftminLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softmin(x)


class SoftminSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return SoftminLayer(ndim), form


class SoftPlusLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softplus(x)


class SoftPlusSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return SoftPlusLayer(ndim), form


class SoftShrinkLayer(TransformLayer):
    def __init__(self, lam, ndim):
        super().__init__(ndim)
        self.lam = lam

    def forward_one(self, x, is_training):
        return Z.softshrink(x, self.lam)


class SoftShrinkSpec(TransformSpec):
    def __init__(self, lam=0.5, ndim=None):
        super().__init__(ndim)
        self.lam = lam

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return SoftShrinkLayer(self.lam, ndim), form


class SoftSignLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softsign(x)


class SoftSignSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return SoftSignLayer(ndim), form


class TanhLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.tanh(x)


class TanhSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return TanhLayer(ndim), form


class TanhShrinkLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.tanh_shrink(x)


class TanhShrinkSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        return TanhShrinkLayer(ndim), form
