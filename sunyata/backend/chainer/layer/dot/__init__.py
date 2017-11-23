from ....base.layer.dot import BaseDotAPI
from .conv import ChainerConvAPI
from .dense import ChainerDenseAPI
from .depthwise_conv import ChainerDepthwiseConvAPI


class ChainerDotAPI(BaseDotAPI, ChainerConvAPI, ChainerDenseAPI,
                    ChainerDepthwiseConvAPI):
    def __init__(self):
        BaseDotAPI.__init__(self)
        ChainerConvAPI.__init__(self)
        ChainerDenseAPI.__init__(self)
        ChainerDepthwiseConvAPI.__init__(self)
