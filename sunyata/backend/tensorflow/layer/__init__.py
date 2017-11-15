from ...base.layer import BaseLayerAPI
from .activation import TensorFlowActivationAPI
from .dot import TensorFlowDotAPI
from .shape import TensorFlowShapeAPI


class TensorFlowLayerAPI(BaseLayerAPI, TensorFlowActivationAPI,
                         TensorFlowDotAPI, TensorFlowShapeAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        TensorFlowActivationAPI.__init__(self)
        TensorFlowDotAPI.__init__(self)
        TensorFlowShapeAPI.__init__(self)
