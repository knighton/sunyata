from ...base.layer import BaseLayerAPI
from .activation import ChainerActivationAPI
from .batch_norm import ChainerBatchNormAPI
from .dot import ChainerDotAPI
from .embed import ChainerEmbedAPI
from .shape import ChainerShapeAPI


class ChainerLayerAPI(BaseLayerAPI, ChainerActivationAPI, ChainerBatchNormAPI,
                      ChainerDotAPI, ChainerEmbedAPI, ChainerShapeAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        ChainerActivationAPI.__init__(self)
        ChainerBatchNormAPI.__init__(self)
        ChainerDotAPI.__init__(self)
        ChainerEmbedAPI.__init__(self)
        ChainerShapeAPI.__init__(self)
