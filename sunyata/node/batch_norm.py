import numpy as np

from .. import backend as Z
from .base import TransformLayer, TransformSpec


class BaseBatchNormLayer(TransformLayer):
    pass


class BaseBatchNormSpec(TransformSpec):
    def _norm_data_shape(self, norm_axis, required_spatial_ndim, batch_shape):
        if required_spatial_ndim is None:
            x_ndim = len(batch_shape) + 1
        else:
            x_ndim = required_spatial_ndim + 2
            assert len(batch_shape) + 1 == x_ndim
        if isinstance(norm_axis, int):
            assert 1 <= norm_axis
            reduction_axes = 0, norm_axis
        elif isinstance(norm_axis, (list, tuple)):
            assert 0 not in norm_axis
            reduction_axes = [0] + sorted(norm_axis)
        else:
            assert False
        shape = [1] + list(batch_shape)
        for axis in reduction_axes:
            shape[axis] = 1
        return shape


class InstanceBatchNormLayer(BaseBatchNormLayer):
    def __init__(self, beta, gamma):
        self.beta = Z.variable(Z.numpy_to_device(beta))
        self.gamma = Z.variable(Z.numpy_to_device(gamma))

    def forward_one(self, x, is_training):
        return Z.instance_batch_norm(x, self.beta, self.gamma)


class InstanceBatchNormSpec(BaseBatchNormSpec):
    def __init__(self, axis=1, ndim=None):
        self.axis = axis
        self.ndim = ndim

    def build_one(self, form):
        shape = self._norm_data_shape(self.axis, self.ndim, form.shape)
        beta = np.zeros(shape, form.dtype)
        gamma = np.ones(shape, form.dtype)
        layer = InstanceBatchNormLayer(beta, gamma)
        return layer, form


class GlobalBatchNormLayer(BaseBatchNormLayer):
    def __init__(self, beta, gamma, momentum, global_mean, global_var):
        self.beta = Z.variable(Z.numpy_to_device(beta))
        self.gamma = Z.variable(Z.numpy_to_device(gamma))
        self.momentum = momentum
        self.global_mean = Z.constant(Z.numpy_to_device(global_mean))
        self.global_var = Z.constant(Z.numpy_to_device(global_var))

    def params(self):
        return [self.beta, self.gamma]

    def forward_one(self, x, is_training):
        return Z.global_batch_norm(
            x, is_training, self.beta, self.gamma, self.momentum,
            self.global_mean, self.global_var)


class GlobalBatchNormSpec(BaseBatchNormSpec):
    def __init__(self, momentum=0.99, axis=1, ndim=None):
        self.momentum = momentum
        self.axis = axis
        self.ndim = ndim

    def build_one(self, form):
        shape = self._norm_data_shape(self.axis, self.ndim, form.shape)
        beta = np.zeros(shape, form.dtype)
        gamma = np.ones(shape, form.dtype)
        global_mean = np.zeros(shape, form.dtype)
        global_var = np.ones(shape, form.dtype)
        layer = GlobalBatchNormLayer(
            beta, gamma, self.momentum, global_mean, global_var)
        return layer, form
