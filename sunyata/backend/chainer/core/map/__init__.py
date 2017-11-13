from ....base.core.map import BaseMapAPI
from .clip import ChainerClipAPI
from .cumulative import ChainerCumulativeAPI
from .hyperbolic import ChainerHyperbolicAPI
from .log_exp import ChainerLogExpAPI
from .power import ChainerPowerAPI
from .round import ChainerRoundAPI
from .sign import ChainerSignAPI
from .trigonometric import ChainerTrigonometricAPI


class ChainerMapAPI(BaseMapAPI, ChainerClipAPI, ChainerCumulativeAPI,
                    ChainerHyperbolicAPI, ChainerLogExpAPI, ChainerPowerAPI,
                    ChainerRoundAPI, ChainerSignAPI, ChainerTrigonometricAPI):
    def __init__(self):
        BaseMapAPI.__init__(self)
        ChainerClipAPI.__init__(self)
        ChainerCumulativeAPI.__init__(self)
        ChainerHyperbolicAPI.__init__(self)
        ChainerLogExpAPI.__init__(self)
        ChainerPowerAPI.__init__(self)
        ChainerRoundAPI.__init__(self)
        ChainerSignAPI.__init__(self)
        ChainerTrigonometricAPI.__init__(self)
