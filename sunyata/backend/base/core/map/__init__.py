from .clip import BaseClipAPI
from .cumulative import BaseCumulativeAPI
from .hyperbolic import BaseHyperbolicAPI
from .log_exp import BaseLogExpAPI
from .power import BasePowerAPI
from .round import BaseRoundAPI
from .sign import BaseSignAPI
from .trigonometric import BaseTrigonometricAPI


class BaseMapAPI(BaseClipAPI, BaseCumulativeAPI, BaseHyperbolicAPI,
                 BaseLogExpAPI, BasePowerAPI, BaseRoundAPI, BaseSignAPI,
                 BaseTrigonometricAPI):
    def __init__(self):
        BaseClipAPI.__init__(self)
        BaseCumulativeAPI.__init__(self)
        BaseHyperbolicAPI.__init__(self)
        BaseLogExpAPI.__init__(self)
        BasePowerAPI.__init__(self)
        BaseRoundAPI.__init__(self)
        BaseSignAPI.__init__(self)
        BaseTrigonometricAPI.__init__(self)
