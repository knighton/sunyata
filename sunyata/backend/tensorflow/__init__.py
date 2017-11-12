import tensorflow.contrib.eager as tfe

from ..base import BaseBackend
from .core import TensorFlowCoreAPI
from .layer import TensorFlowLayerAPI


tfe.enable_eager_execution()


class TensorFlowBackend(BaseBackend, TensorFlowCoreAPI, TensorFlowLayerAPI):
    def __init__(self):
        BaseBackend.__init__(self)
        TensorFlowCoreAPI.__init__(self)
        TensorFlowLayerAPI.__init__(self)
        self.name = 'tensorflow'
