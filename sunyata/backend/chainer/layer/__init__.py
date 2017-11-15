from ...base.layer import BaseLayerAPI
from .activation import ChainerActivationAPI
from .dot import ChainerDotAPI


class ChainerLayerAPI(BaseLayerAPI, ChainerActivationAPI, ChainerDotAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        ChainerActivationAPI.__init__(self)
        ChainerDotAPI.__init__(self)
