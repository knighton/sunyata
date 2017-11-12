from ..base import APIBase


class BaseActivationAPI(APIBase):
    def __init__(self):
        APIBase.__init__(self)

    def relu(self, x):
        return self.clip(x, min=0)

    def softmax(self, x):
        raise NotImplementedError
