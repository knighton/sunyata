from .activation import BaseActivationAPI
from .dot import BaseDotAPI
from .shape import BaseShapeAPI


class BaseLayerAPI(BaseActivationAPI, BaseDotAPI, BaseShapeAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)
        BaseDotAPI.__init__(self)
        BaseShapeAPI.__init__(self)
