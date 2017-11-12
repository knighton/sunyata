from ..base import APIMixin


class BaseAccuracyAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def categorical_accuracy(self, true, pred):
        true_indices = self.argmax(true, -1)
        pred_indices = self.argmax(pred, -1)
        hits = self.equal(true_indices, pred_indices)
        hits = self.cast(hits, self.dtype_of(true))
        return self.mean(hits, -1, False)
