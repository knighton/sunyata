from .activation import BaseActivationAPI
from .batch_norm import BaseBatchNormAPI
from .dot import BaseDotAPI
from .noise import BaseNoiseAPI
from .shape import BaseShapeAPI


class BaseLayerAPI(BaseActivationAPI, BaseBatchNormAPI, BaseDotAPI,
                   BaseNoiseAPI, BaseShapeAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)
        BaseBatchNormAPI.__init__(self)
        BaseDotAPI.__init__(self)
        BaseNoiseAPI.__init__(self)
        BaseShapeAPI.__init__(self)
