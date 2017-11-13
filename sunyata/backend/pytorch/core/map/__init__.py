from ....base.core.map import BaseMapAPI
from .clip import PyTorchClipAPI
from .cumulative import PyTorchCumulativeAPI
from .hyperbolic import PyTorchHyperbolicAPI
from .log_exp import PyTorchLogExpAPI
from .power import PyTorchPowerAPI
from .round import PyTorchRoundAPI
from .sign import PyTorchSignAPI
from .trigonometric import PyTorchTrigonometricAPI


class PyTorchMapAPI(BaseMapAPI, PyTorchClipAPI, PyTorchCumulativeAPI,
                    PyTorchHyperbolicAPI, PyTorchLogExpAPI, PyTorchPowerAPI,
                    PyTorchRoundAPI, PyTorchSignAPI, PyTorchTrigonometricAPI):
    def __init__(self):
        BaseMapAPI.__init__(self)
        PyTorchClipAPI.__init__(self)
        PyTorchCumulativeAPI.__init__(self)
        PyTorchHyperbolicAPI.__init__(self)
        PyTorchLogExpAPI.__init__(self)
        PyTorchPowerAPI.__init__(self)
        PyTorchRoundAPI.__init__(self)
        PyTorchSignAPI.__init__(self)
        PyTorchTrigonometricAPI.__init__(self)
