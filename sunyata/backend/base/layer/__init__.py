from .activation import BaseActivationAPI
from .batch_norm import BatchNormAPI
from .dot import BaseDotAPI
from .shape import BaseShapeAPI


class BaseLayerAPI(BaseActivationAPI, BaseDotAPI, BatchNormAPI, BaseShapeAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)
        BaseBatchNormAPI.__init__(self)
        BaseDotAPI.__init__(self)
        BaseShapeAPI.__init__(self)
