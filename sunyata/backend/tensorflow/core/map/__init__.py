from ....base.core.map import BaseMapAPI
from .clip import TensorFlowClipAPI
from .cumulative import TensorFlowCumulativeAPI
from .hyperbolic import TensorFlowHyperbolicAPI
from .log_exp import TensorFlowLogExpAPI
from .power import TensorFlowPowerAPI
from .round import TensorFlowRoundAPI
from .sign import TensorFlowSignAPI
from .trigonometric import TensorFlowTrigonometricAPI


class TensorFlowMapAPI(BaseMapAPI, TensorFlowClipAPI, TensorFlowCumulativeAPI,
                       TensorFlowHyperbolicAPI, TensorFlowLogExpAPI,
                       TensorFlowPowerAPI, TensorFlowRoundAPI,
                       TensorFlowSignAPI, TensorFlowTrigonometricAPI):
    def __init__(self):
        BaseMapAPI.__init__(self)
        TensorFlowClipAPI.__init__(self)
        TensorFlowCumulativeAPI.__init__(self)
        TensorFlowHyperbolicAPI.__init__(self)
        TensorFlowLogExpAPI.__init__(self)
        TensorFlowPowerAPI.__init__(self)
        TensorFlowRoundAPI.__init__(self)
        TensorFlowSignAPI.__init__(self)
        TensorFlowTrigonometricAPI.__init__(self)
