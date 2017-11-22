from .. import backend as Z
from .base import TransformLayer, TransformSpec


class ArcTanLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.arctan(x)


class ArcTanSpec(TransformSpec):
    def build_one(self, form):
        return ArcTanLayer(), form


class BentIdentityLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.bent_identity(x)


class BentIdentitySpec(TransformSpec):
    def build_one(self, form):
        return BentIdentityLayer(), form


class ELULayer(TransformLayer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward_one(self, x, is_training):
        return Z.elu(x, self.alpha)


class ELUSpec(TransformSpec):
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def build_one(self, form):
        return ELULayer(self.alpha), form


class HardShrinkLayer(TransformLayer):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def forward_one(self, x, is_training):
        return Z.hard_shrink(x, self.lam)


class HardShrinkSpec(TransformSpec):
    def __init__(self, lam=0.5):
        super().__init__()
        self.lam = lam

    def build_one(self, form):
        return HardShrinkLayer(self.lam), form


class HardSigmoidLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.hard_sigmoid(x)


class HardSigmoidSpec(TransformSpec):
    def build_one(self, form):
        return HardSigmoidLayer(), form


class HardTanhLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.hard_tanh(x)


class HardTanhSpec(TransformSpec):
    def build_one(self, form):
        return HardTanhLayer(), form


class IdentityLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.identity(x)


class IdentitySpec(TransformSpec):
    def build_one(self, form):
        return IdentityLayer(), form


class LeakyReLULayer(TransformLayer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward_one(self, x, is_training):
        return Z.leaky_relu(x, self.alpha)


class LeakyReLUSpec(TransformSpec):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def build_one(self, form):
        return LeakyReLULayer(self.alpha), form


class LogSigmoidLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.log_sigmoid(x)


class LogSigmoidSpec(TransformSpec):
    def build_one(self, form):
        return LogSigmoidLayer(), form


class ReLULayer(TransformLayer):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward_one(self, x, is_training):
        return Z.relu(x, self.min, self.max)


class ReLUSpec(TransformSpec):
    def __init__(self, min=0., max=None):
        super().__init__()
        self.min = min
        self.max = max

    def build_one(self, form):
        return ReLULayer(self.min, self.max), form


class SELULayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.selu(x)


class SELUSpec(TransformSpec):
    def build_one(self, form):
        return SELULayer(), form


class SigmoidLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.sigmoid(x)


class SigmoidSpec(TransformSpec):
    def build_one(self, form):
        return SigmoidLayer(), form


class SoftExponentialLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softexponential(x)


class SoftExponentialSpec(TransformSpec):
    def build_one(self, form):
        return SoftExponentialLayer(), form


class SoftmaxLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softmax(x)


class SoftmaxSpec(TransformSpec):
    def build_one(self, form):
        return SoftmaxLayer(), form


class SoftminLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softmin(x)


class SoftminSpec(TransformSpec):
    def build_one(self, form):
        return SoftminLayer(), form


class SoftPlusLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softplus(x)


class SoftPlusSpec(TransformSpec):
    def build_one(self, form):
        return SoftPlusLayer(), form


class SoftShrinkLayer(TransformLayer):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def forward_one(self, x, is_training):
        return Z.softshrink(x, self.lam)


class SoftShrinkSpec(TransformSpec):
    def __init__(self, lam=0.5):
        super().__init__()
        self.lam = lam

    def build_one(self, form):
        return SoftShrinkLayer(self.lam), form


class SoftSignLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.softsign(x)


class SoftSignSpec(TransformSpec):
    def build_one(self, form):
        return SoftSignLayer(), form


class TanhLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.tanh(x)


class TanhSpec(TransformSpec):
    def build_one(self, form):
        return TanhLayer(), form


class TanhShrinkLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.tanh_shrink(x)


class TanhShrinkSpec(TransformSpec):
    def build_one(self, form):
        return TanhShrinkLayer(), form
