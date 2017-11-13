import numpy as np

from ....base.core.map.clip import BaseClipAPI


class PyTorchClipAPI(BaseClipAPI):
    def __init__(self):
        BaseClipAPI.__init__(self)

    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)
