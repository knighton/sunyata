from ...base.layer import BaseLayerAPI
from .activation import TensorFlowActivationAPI
from .dot import TensorFlowDotAPI


class TensorFlowLayerAPI(BaseLayerAPI, TensorFlowActivationAPI,
                         TensorFlowDotAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        TensorFlowActivationAPI.__init__(self)
        TensorFlowDotAPI.__init__(self)
