from .alpha_dropout import BaseAlphaDropoutAPI
from .dropout import BaseDropoutAPI
from .gaussian_dropout import BaseGaussianDropoutAPI
from .gaussian_noise import BaseGaussianNoiseAPI


class BaseNoiseAPI(BaseAlphaDropoutAPI, BaseDropoutAPI, BaseGaussianDropoutAPI,
                   BaseGaussianNoiseAPI):
    def __init__(self):
        BaseAlphaDropoutAPI.__init__(self)
        BaseDropoutAPI.__init__(self)
        BaseGaussianDropoutAPI.__init__(self)
        BaseGaussianNoiseAPI.__init__(self)
