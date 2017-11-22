from ..base import APIMixin


class BaseBatchNormAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def do_batch_norm(self, x, beta, gamma, mean, var):
        x = (x - mean) / self.sqrt(var + self.epsilon())
        if gamma is not None:
            x *= gamma
        if beta is not None:
            x += beta
        return x

    def _instance_batch_norm(self, x, beta, gamma, ndim):
        if ndim is None:
            ndim = self.ndim(x) - 2
        else:
            assert self.ndim(x) - 2 == ndim
        reduce_axes = [0] + list(range(2, ndim))
        mean, var = self.moments(x, reduce_axes)
        return self.do_batch_norm(x, beta, gamma, mean, var)

    def instance_batch_norm(self, x, beta, gamma):
        return self._instance_batch_norm(x, beta, gamma, None)

    def instance_batch_norm0d(self, x, beta, gamma):
        return self._instance_batch_norm(x, beta, gamma, 0)

    def instance_batch_norm1d(self, x, beta, gamma):
        return self._instance_batch_norm(x, beta, gamma, 1)

    def instance_batch_norm2d(self, x, beta, gamma):
        return self._instance_batch_norm(x, beta, gamma, 2)

    def instance_batch_norm3d(self, x, beta, gamma):
        return self._instance_batch_norm(x, beta, gamma, 3)

    def _global_batch_norm(self, x, is_training, beta, gamma, momentum,
                           moving_mean, moving_var, ndim):
        if ndim is None:
            ndim = self.ndim(x) - 2
        else:
            assert self.ndim(x) - 2 == ndim
        if is_training:
            reduce_axes = [0] + list(range(2, ndim + 2))
            mean, var = self.moments(x, reduce_axes)
            x = self.do_batch_norm(x, beta, gamma, mean, var)
            self.assign_momentum(moving_mean, mean, momentum)
            self.assign_momentum(moving_var, mean, momentum)
        else:
            x = self.do_batch_norm(x, beta, gamma, moving_mean, moving_var)
        return x

    def global_batch_norm(self, x, is_training, beta, gamma, momentum,
                          moving_mean, moving_var):
        return self._global_batch_norm(x, is_training, beta, gamma, momentum,
                                       moving_mean, moving_var, None)

    def global_batch_norm0d(self, x, is_training, beta, gamma, momentum,
                            moving_mean, moving_var):
        return self._global_batch_norm(x, is_training, beta, gamma, momentum,
                                       moving_mean, moving_var, 0)

    def global_batch_norm1d(self, x, is_training, beta, gamma, momentum,
                            moving_mean, moving_var):
        return self._global_batch_norm(x, is_training, beta, gamma, momentum,
                                       moving_mean, moving_var, 1)

    def global_batch_norm2d(self, x, is_training, beta, gamma, momentum,
                            moving_mean, moving_var):
        return self._global_batch_norm(x, is_training, beta, gamma, momentum,
                                       moving_mean, moving_var, 2)

    def global_batch_norm3d(self, x, is_training, beta, gamma, momentum,
                            moving_mean, moving_var):
        return self._global_batch_norm(x, is_training, beta, gamma, momentum,
                                       moving_mean, moving_var, 3)
