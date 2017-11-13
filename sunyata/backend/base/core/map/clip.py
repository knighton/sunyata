import numpy as np

from ...base import APIMixin


class BaseClipAPI(APIMixin):
    def clip(self, x, min=-np.inf, max=np.inf):
        raise NotImplementedError
