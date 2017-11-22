from .activation import BaseActivationAPI
from .alpha_dropout import BaseAlphaDropoutAPI
from .batch_norm import BaseBatchNormAPI
from .dot import BaseDotAPI
from .dropout import BaseDropoutAPI
from .gaussian_dropout import BaseGaussianDropoutAPI
from .shape import BaseShapeAPI


class BaseLayerAPI(BaseActivationAPI, BaseAlphaDropoutAPI, BaseBatchNormAPI,
                   BaseDotAPI, BaseDropoutAPI, BaseGaussianDropoutAPI,
                   BaseShapeAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)
        BaseAlphaDropoutAPI.__init__(self)
        BaseBatchNormAPI.__init__(self)
        BaseDotAPI.__init__(self)
        BaseDropoutAPI.__init__(self)
        BaseGaussianDropoutAPI.__init__(self)
        BaseShapeAPI.__init__(self)
