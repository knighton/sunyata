from ...base.layer import BaseLayerAPI
from .activation import ChainerActivationAPI
from .dot import ChainerDotAPI
from .shape import ChainerShapeAPI


class ChainerLayerAPI(BaseLayerAPI, ChainerActivationAPI, ChainerDotAPI,
                      ChainerShapeAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        ChainerActivationAPI.__init__(self)
        ChainerDotAPI.__init__(self)
        ChainerShapeAPI.__init__(self)
