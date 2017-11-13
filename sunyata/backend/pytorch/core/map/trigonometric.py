from ....base.core.map.trigonometric import BaseTrigonometricAPI


class PyTorchTrigonometricAPI(BaseTrigonometricAPI):
    def __init__(self):
        BaseTrigonometricAPI.__init__(self)

    def sin(self, x):
        return x.sin()

    def cos(self, x):
        return x.cos()

    def tan(self, x):
        return x.tan()

    def arcsin(self, x):
        return x.asin()

    def arccos(self, x):
        return x.acos()

    def arctan(self, x):
        return x.atan()
