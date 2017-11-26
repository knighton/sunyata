import numpy as np
from scipy.stats import truncnorm

from .base import Initializer


def _get_min_stds(dist_mean, std, min_stds=None, min_value=None,
                  default_min_stds=-2):
    if min_stds is None:
        if min_value is None:
            stds = default_min_stds
        else:
            stds = (min_value - dist_mean) / std
    else:
        if min_value is None:
            stds = min_stds
        else:
            assert False
    return stds


def _get_max_stds(dist_mean, std, max_stds=None, max_value=None,
                  default_max_stds=2):
    if max_stds is None:
        if max_value is None:
            stds = default_max_stds
        else:
            stds = (max_value - dist_mean) / std
    else:
        if max_value is None:
            stds = max_stds
        else:
            assert False
    return stds


def _trunc_norm_stds(shape, mean=0, std=1, dtype='float32', min_stds=-2,
                     max_stds=2):
    dist = truncnorm(min_stds, max_stds)
    x = dist.rvs(np.prod(shape)).reshape(shape)
    x = x * std + mean
    return x.astype(dtype)


def _trunc_norm(shape, mean=0, std=1, dtype='float32', min_stds=None,
                max_stds=None, min_value=None, max_value=None,
                default_min_stds=-2, default_max_stds=2):
    min_stds = _get_min_stds(mean, std, min_stds, min_value, default_min_stds)
    max_stds = _get_max_stds(mean, std, max_stds, max_value, default_max_stds)
    return _trunc_norm_stds(shape, mean, std, dtype, min_stds, max_stds)


class TruncNorm(Initializer):
    def __init__(self, mean=0, std=1, min_stds=None, max_stds=None,
                 min_value=None, max_value=None, default_min_stds=-2,
                 default_max_stds=2):
        self.mean = mean
        self.std = std
        self.min_stds = min_stds
        self.max_stds = max_stds
        self.min_value = min_value
        self.max_value = max_value
        self.default_min_stds = default_min_stds
        self.default_max_stds = default_max_stds

    def __call__(self, shape, dtype='float32', meaning=None):
        return _trunc_norm(shape, self.mean, self.std, dtype, self.min_stds,
                           self.max_stds, self.min_value, self.max_value,
                           self.default_min_stds, self.default_max_stds)


trunc_norm = TruncNorm
