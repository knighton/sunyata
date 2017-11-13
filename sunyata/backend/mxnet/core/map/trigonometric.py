import mxnet as mx

from ....base.core.map.trigonometric import BaseTrigonometricAPI


class MXNetTrigonometricAPI(BaseTrigonometricAPI):
    def __init__(self):
        BaseTrigonometricAPI.__init__(self)

    def sin(self, x):
        return mx.nd.sin(x)

    def cos(self, x):
        return mx.nd.cos(x)

    def tan(self, x):
        return mx.nd.tan(x)

    def arcsin(self, x):
        return mx.nd.arcsin(x)

    def arccos(self, x):
        return mx.nd.arccos(x)

    def arctan(self, x):
        return mx.nd.arctan(x)
