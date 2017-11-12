from ..base import APIBase


class BaseActivationAPI(APIBase):
    def relu(self, x):
        return self.clip(x, min=0)

    def softmax(self, x):
        raise NotImplementedError
