from ...base.layer import BaseLayerAPI
from .activation import MXNetActivationAPI
from .dot import MXNetDotAPI
from .embed import MXNetEmbedAPI
from .shape import MXNetShapeAPI


class MXNetLayerAPI(BaseLayerAPI, MXNetActivationAPI, MXNetDotAPI,
                     MXNetEmbedAPI, MXNetShapeAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        MXNetActivationAPI.__init__(self)
        MXNetDotAPI.__init__(self)
        MXNetEmbedAPI.__init__(self)
        MXNetShapeAPI.__init__(self)
