from .conv import BaseConvAPI
from .dense import BaseDenseAPI
from .depthwise_conv import BaseDepthwiseConvAPI
from .separable_conv import BaseSeparableConvAPI


class BaseDotAPI(BaseConvAPI, BaseDenseAPI, BaseSeparableConvAPI):
    def __init__(self):
        BaseConvAPI.__init__(self)
        BaseDenseAPI.__init__(self)
        BaseDepthwiseConvAPI.__init__(self)
        BaseSeparableConvAPI.__init__(self)
