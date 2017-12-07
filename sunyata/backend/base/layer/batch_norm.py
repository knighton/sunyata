from ..base import APIMixin


class BaseBatchNormAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def do_batch_norm(self, x, beta, gamma, mean, var):
        return gamma * (x - mean) / self.sqrt(var + self.epsilon()) + beta

    def _batch_norm_reduction_axes(self, shape):
        axes = []
        for i, dim in enumerate(shape):
            if dim == 1:
                axes.append(i)
        return axes

    def instance_batch_norm(self, x, beta, gamma):
        reduction_axes = self._batch_norm_reduction_axes(self.shape(beta))
        instance_mean, instance_var = self.moments(x, reduction_axes)
        return self.do_batch_norm(x, beta, gamma, instance_mean, instance_var)

    def instance_batch_norm0d(self, x, beta, gamma):
        return self.instance_batch_norm(x, beta, gamma)

    def instance_batch_norm1d(self, x, beta, gamma):
        return self.instance_batch_norm(x, beta, gamma)

    def instance_batch_norm2d(self, x, beta, gamma):
        return self.instance_batch_norm(x, beta, gamma)

    def instance_batch_norm3d(self, x, beta, gamma):
        return self.instance_batch_norm(x, beta, gamma)

    def global_batch_norm(self, x, train, beta, gamma, momentum, global_mean,
                          global_var):
        if train:
            reduction_axes = self._batch_norm_reduction_axes(self.shape(beta))
            instance_mean, instance_var = self.moments(x, reduction_axes)
            x = self.do_batch_norm(x, beta, gamma, instance_mean, instance_var)
            self.assign_momentum(global_mean, instance_mean, momentum)
            self.assign_momentum(global_var, instance_var, momentum)
        else:
            x = self.do_batch_norm(x, beta, gamma, global_mean, global_var)
        return x

    def global_batch_norm0d(self, x, train, beta, gamma, momentum, global_mean,
                            global_var):
        return self.global_batch_norm(
            x, train, beta, gamma, momentum, global_mean, global_var)

    def global_batch_norm1d(self, x, train, beta, gamma, momentum, global_mean,
                            global_var):
        return self.global_batch_norm(
            x, train, beta, gamma, momentum, global_mean, global_var)

    def global_batch_norm2d(self, x, train, beta, gamma, momentum, global_mean,
                            global_var):
        return self.global_batch_norm(
            x, train, beta, gamma, momentum, global_mean, global_var)

    def global_batch_norm3d(self, x, train, beta, gamma, momentum, global_mean,
                            global_var):
        return self.global_batch_norm(
            x, train, beta, gamma, momentum, global_mean, global_var)
