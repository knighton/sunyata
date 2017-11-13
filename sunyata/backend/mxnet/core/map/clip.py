import mxnet as mx
import numpy as np

from ....base.core.map.clip import BaseClipAPI


class MXNetClipAPI(BaseClipAPI):
    def __init__(self):
        BaseClipAPI.__init__(self)

    def clip(self, x, min=-np.inf, max=np.inf):
        return mx.nd.clip(x, min, max)
