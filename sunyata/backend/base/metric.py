from .base import APIBase


class BaseMetricAPI(APIBase):
    def binary_cross_entropy(self, true, pred):
        pred = self.clip(pred, self.epsilon(), 1 - self.epsilon())
        return -true * self.log(pred) - (1 - true) * self.log(1 - pred)

    def categorical_cross_entropy(self, true, pred):
        pred = self.clip(pred, self.epsilon(), 1 - self.epsilon())
        return self.mean(-true * self.log(pred), -1)

    def mean_squared_error(self, true, pred):
        return self.mean(self.pow(true - pred, 2), -1)

    def categorical_accuracy(self, true, pred):
        true_indices = self.argmax(true, -1)
        pred_indices = self.argmax(pred, -1)
        hits = self.equal(true_indices, pred_indices)
        hits = self.cast(hits, self.dtype_of(true))
        return self.mean(hits, -1, False)
