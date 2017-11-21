from .....base.layer.shape.upsample import BaseUpsampleAPI
from .linear import TensorFlowLinearUpsampleAPI
from .nearest import TensorFlowNearestUpsampleAPI


class TensorFlowUpsampleAPI(BaseUpsampleAPI, TensorFlowLinearUpsampleAPI,
                            TensorFlowNearestUpsampleAPI):
    def __init__(self):
        TensorFlowLinearUpsampleAPI.__init__(self)
        TensorFlowNearestUpsampleAPI.__init__(self)
