from .. import backend as Z
from .base import Metric


class Loss(Metric):
    pass


class BinaryCrossEntropy(Loss):
    def __call__(self, true, pred):
        return Z.binary_cross_entropy(true, pred)


class CategoricalCrossEntropy(Loss):
    def __call__(self, true, pred):
        return Z.categorical_cross_entropy(true, pred)


class MeanSquaredError(Loss):
    def __call__(self, true, pred):
        return Z.mean_squared_error(true, pred)
