from ..base import APIBase


class BaseLossAPI(APIBase):
    def binary_cross_entropy(self, true, pred):
        pred = self.clip(pred, self.epsilon(), 1 - self.epsilon())
        return -true * self.log(pred) - (1 - true) * self.log(1 - pred)

    def categorical_cross_entropy(self, true, pred):
        pred = self.clip(pred, self.epsilon(), 1 - self.epsilon())
        return self.mean(-true * self.log(pred), -1)

    def mean_squared_error(self, true, pred):
        return self.mean(self.pow(true - pred, 2), -1)
