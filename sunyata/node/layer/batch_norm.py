from ... import backend as Z
from ... import init
from .base import node_wrap, TransformLayer, TransformSpec


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
        super().__init__()
        self.beta = self.add_param(beta)
        self.gamma = self.add_param(gamma)

    def transform(self, x, train):
        return Z.instance_batch_norm(x, self.beta, self.gamma)


class InstanceBatchNormSpec(BaseBatchNormSpec):
    def __init__(self, axis=1, beta_init='zeros', gamma_init='ones',
                 spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.axis = axis
        self.beta_init = init.get(beta_init)
        self.gamma_init = init.get(gamma_init)

    def build_transform(self, form):
        shape = self._norm_data_shape(
            self.axis, self.spatial_ndim(), form.batch_shape)
        beta = self.beta_init(shape, form.dtype)
        gamma = self.gamma_init(shape, form.dtype)
        layer = InstanceBatchNormLayer(beta, gamma)
        return layer, form


node_wrap(InstanceBatchNormSpec)


class GlobalBatchNormLayer(BaseBatchNormLayer):
    def __init__(self, beta, gamma, momentum, mean, var, x_ndim):
        super().__init__(x_ndim)
        self.beta = self.add_param(beta)
        self.gamma = self.add_param(gamma)
        self.momentum = momentum
        self.mean = self.add_param(mean, trainable=False)
        self.var = self.add_param(var, trainable=False)

    def transform(self, x, train):
        return Z.global_batch_norm(
            x, train, self.beta, self.gamma, self.momentum, self.mean,
            self.var)


class GlobalBatchNormSpec(BaseBatchNormSpec):
    def __init__(self, momentum=0.99, axis=1, beta_init='zeros',
                 gamma_init='ones', mean_init='zeros', var_init='ones',
                 spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.momentum = momentum
        self.axis = axis
        self.beta_init = init.get(beta_init)
        self.gamma_init = init.get(gamma_init)
        self.mean_init = init.get(mean_init)
        self.var_init = init.get(var_init)

    def build_transform(self, form):
        shape = self._norm_data_shape(
            self.axis, self.spatial_ndim(), form.batch_shape)
        beta = self.beta_init(shape, form.dtype)
        gamma = self.gamma_init(shape, form.dtype)
        mean = self.mean_init(shape, form.dtype)
        var = self.var_init(shape, form.dtype)
        layer = GlobalBatchNormLayer(
            beta, gamma, self.momentum, mean, var, self.x_ndim())
        return layer, form


node_wrap(GlobalBatchNormSpec)
