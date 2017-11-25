from ...base.layer import BaseLayerAPI
from .activation import TensorFlowActivationAPI
from .dot import TensorFlowDotAPI
from .embed import TensorFlowEmbedAPI
from .shape import TensorFlowShapeAPI


class TensorFlowLayerAPI(BaseLayerAPI, TensorFlowActivationAPI,
                         TensorFlowDotAPI, TensorFlowEmbedAPI,
                         TensorFlowShapeAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        TensorFlowActivationAPI.__init__(self)
        TensorFlowDotAPI.__init__(self)
        TensorFlowEmbedAPI.__init__(self)
        TensorFlowShapeAPI.__init__(self)
