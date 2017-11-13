from chainer import functions as F
import numpy as np

from ....base.core.map.clip import BaseClipAPI


class ChainerClipAPI(BaseClipAPI):
    def __init__(self):
        BaseClipAPI.__init__(self)

    def clip(self, x, min=-np.inf, max=np.inf):
        return F.clip(x, float(min), float(max))
