import numpy as np

from ... import backend as Z
from .base import node_wrap, TransformLayer, TransformSpec


class ArcTanLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.arctan(x)


class ArcTanSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return ArcTanLayer(), form


node_wrap(ArcTanSpec)


class BentIdentityLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.bent_identity(x)


class BentIdentitySpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return BentIdentityLayer(), form


node_wrap(BentIdentitySpec)


class ELULayer(TransformLayer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def transform(self, x, is_training):
        return Z.elu(x, self.alpha)


class ELUSpec(TransformSpec):
    def __init__(self, alpha=1., ndim=None):
        super().__init__(ndim)
        self.alpha = alpha

    def build_transform(self, form):
        return ELULayer(self.alpha), form


node_wrap(ELUSpec)


class HardShrinkLayer(TransformLayer):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def transform(self, x, is_training):
        return Z.hard_shrink(x, self.lam)


class HardShrinkSpec(TransformSpec):
    def __init__(self, lam=0.5, ndim=None):
        super().__init__(ndim)
        self.lam = lam

    def build_transform(self, form):
        return HardShrinkLayer(self.lam), form


node_wrap(HardShrinkSpec)


class HardSigmoidLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.hard_sigmoid(x)


class HardSigmoidSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return HardSigmoidLayer(), form


node_wrap(HardSigmoidSpec)


class HardTanhLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.hard_tanh(x)


class HardTanhSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return HardTanhLayer(), form


node_wrap(HardTanhSpec)


class IdentityLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.identity(x)


class IdentitySpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return IdentityLayer(), form


node_wrap(IdentitySpec)


class LeakyReLULayer(TransformLayer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def transform(self, x, is_training):
        return Z.leaky_relu(x, self.alpha)


class LeakyReLUSpec(TransformSpec):
    def __init__(self, alpha=0.1, ndim=None):
        super().__init__(ndim)
        self.alpha = alpha

    def build_transform(self, form):
        return LeakyReLULayer(self.alpha), form


node_wrap(LeakyReLUSpec)


class LogSigmoidLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.log_sigmoid(x)


class LogSigmoidSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return LogSigmoidLayer(), form


node_wrap(LogSigmoidSpec)


class ReLULayer(TransformLayer):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def transform(self, x, is_training):
        return Z.relu(x, self.min, self.max)


class ReLUSpec(TransformSpec):
    def __init__(self, min=0., max=np.inf, ndim=None):
        super().__init__(ndim)
        self.min = min
        self.max = max

    def build_transform(self, form):
        return ReLULayer(self.min, self.max), form


node_wrap(ReLUSpec)


class SELULayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.selu(x)


class SELUSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SELULayer(), form


node_wrap(SELUSpec)


class SigmoidLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.sigmoid(x)


class SigmoidSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SigmoidLayer(), form


node_wrap(SigmoidSpec)


class SoftExponentialLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softexponential(x)


class SoftExponentialSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftExponentialLayer(), form


node_wrap(SoftExponentialSpec)


class SoftmaxLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softmax(x)


class SoftmaxSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftmaxLayer(), form


node_wrap(SoftmaxSpec)


class SoftminLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softmin(x)


class SoftminSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftminLayer(), form


node_wrap(SoftminSpec)


class SoftPlusLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softplus(x)


class SoftPlusSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftPlusLayer(), form


node_wrap(SoftPlusSpec)


class SoftShrinkLayer(TransformLayer):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def transform(self, x, is_training):
        return Z.softshrink(x, self.lam)


class SoftShrinkSpec(TransformSpec):
    def __init__(self, lam=0.5, ndim=None):
        super().__init__(ndim)
        self.lam = lam

    def build_transform(self, form):
        return SoftShrinkLayer(self.lam), form


node_wrap(SoftShrinkSpec)


class SoftSignLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.softsign(x)


class SoftSignSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return SoftSignLayer(), form


node_wrap(SoftSignSpec)


class TanhLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.tanh(x)


class TanhSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return TanhLayer(), form


node_wrap(TanhSpec)


class TanhShrinkLayer(TransformLayer):
    def transform(self, x, is_training):
        return Z.tanh_shrink(x)


class TanhShrinkSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_transform(self, form):
        return TanhShrinkLayer(), form


node_wrap(TanhShrinkSpec)
