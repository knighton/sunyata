import numpy as np

from .base import Initializer
from .normal import _normal
from .truncated_normal import _truncated_normal_stds
from .uniform import _uniform


def _get_fans(shape, meaning):
    if meaning == 'conv_kernel':
        fan = int(np.prod(shape[2:]))
        fan_in = fan * shape[1]
        fan_out = fan * shape[0]
    else:
        assert False
    return fan_in, fan_out


def _weight_fans(fan_mode, fan_in, fan_out):
    if fan_mode == 'avg':
        fan = (fan_in + fan_out) / 2
    elif fan_mode == 'in':
        fan = fan_in
    elif fan_mode == 'out':
        fan = fan_out
    else:
        assert False
    return fan


def _smart_scaler_fan(shape, dtype, dist, fan, scale=1):
    scale /= fan
    if dist == 'normal':
        std = np.sqrt(scale)
        x = _normal(shape, 0, std, dtype)
    elif dist == 'truncated_normal':
        std = np.sqrt(scale)
        x = _truncated_normal_stds(shape, 0, std, dtype)
    elif dist == 'uniform':
        limit = np.sqrt(3 * scale)
        x = _uniform(shape, -limit, limit, dtype)
    return x


def _smart_scaler(shape, meaning, dtype, dist, fan_mode, scale=1):
    fan_in, fan_out = _get_fans(shape, meaning)
    weighted_fan = _weight_fans(fan_mode, fan_in, fan_out)
    return _smart_scaler_fan(shape, dtype, dist, weighted_fan, scale)


class SmartScaler(Initializer):
    def __init__(self, dist, fan, scale=1):
        self.dist = dist
        self.fan_mode = fan
        self.scale = scale

    def __call__(self, shape, dtype='float32', meaning=None):
        return _smart_scaler(shape, meaning, dtype, self.dist, self.fan_mode,
                             self.scale)


smart_scaler = SmartScaler


def glorot_normal():
    return SmartScaler('normal', 'avg', 1)


def glorot_truncated_normal():
    return SmartScaler('truncated_normal', 'avg', 1)


def glorot_uniform():
    return SmartScaler('uniform', 'avg', 1)


def he_normal():
    return SmartScaler('normal', 'in', 2)


def he_truncated_normal():
    return SmartScaler('truncated_normal', 'in', 2)


def he_uniform():
    return SmartScaler('uniform', 'in', 2)


def lecun_normal():
    return SmartScaler('normal', 'in', 1)


def lecun_truncated_normal():
    return SmartScaler('truncated_normal', 'in', 1)


def lecun_uniform():
    return SmartScaler('uniform', 'in', 1)
