from ....base.core.map import BaseMapAPI
from .clip import MXNetClipAPI
from .cumulative import MXNetCumulativeAPI
from .hyperbolic import MXNetHyperbolicAPI
from .log_exp import MXNetLogExpAPI
from .power import MXNetPowerAPI
from .round import MXNetRoundAPI
from .sign import MXNetSignAPI
from .trigonometric import MXNetTrigonometricAPI


class MXNetMapAPI(BaseMapAPI, MXNetClipAPI, MXNetCumulativeAPI,
                  MXNetHyperbolicAPI, MXNetLogExpAPI, MXNetPowerAPI,
                  MXNetRoundAPI, MXNetSignAPI, MXNetTrigonometricAPI):
    def __init__(self):
        BaseMapAPI.__init__(self)
        MXNetClipAPI.__init__(self)
        MXNetCumulativeAPI.__init__(self)
        MXNetHyperbolicAPI.__init__(self)
        MXNetLogExpAPI.__init__(self)
        MXNetPowerAPI.__init__(self)
        MXNetRoundAPI.__init__(self)
        MXNetSignAPI.__init__(self)
        MXNetTrigonometricAPI.__init__(self)
