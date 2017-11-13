from chainer import functions as F

from ....base.core.map.trigonometric import BaseTrigonometricAPI


class ChainerTrigonometricAPI(BaseTrigonometricAPI):
    def __init__(self):
        BaseTrigonometricAPI.__init__(self)

    def sin(self, x):
        return F.sin(x)

    def cos(self, x):
        return F.cos(x)

    def tan(self, x):
        return F.tan(x)

    def arcsin(self, x):
        return F.arcsin(x)

    def arccos(self, x):
        return F.arccos(x)

    def arctan(self, x):
        return F.arctan(x)
