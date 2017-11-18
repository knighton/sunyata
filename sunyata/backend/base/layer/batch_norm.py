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

    def instance_batch_norm(self, x, beta, gamma):
        axes = [0] + list(range(2, self.ndim(x)))
        mean, var = self.moments(x, axes)
        return self.do_batch_norm(x, beta, gamma, mean, var)

    def instance_batch_norm0d(self, x, beta, gamma):
        assert self.ndim(x) == 2
        return self.instance_batch_norm(x, beta, gamma)

    def instance_batch_norm1d(self, x, beta, gamma):
        assert self.ndim(x) == 3
        return self.instance_batch_norm(x, beta, gamma)

    def instance_batch_norm2d(self, x, beta, gamma):
        assert self.ndim(x) == 4
        return self.instance_batch_norm(x, beta, gamma)

    def instance_batch_norm3d(self, x, beta, gamma):
        assert self.ndim(x) == 5
        return self.instance_batch_norm(x, beta, gamma)

    def global_batch_norm(self, x, is_training, beta, gamma, momentum,
                          moving_mean, moving_var, ndim=None):
        if is_training:
            axes = [0] + list(range(2, self.ndim(x)))
            mean, var = self.moments(x, axes)
            x = self.do_batch_norm(x, beta, gamma, mean, var)
            self.assign_momentum(moving_mean, mean, momentum)
            self.assign_momentum(moving_var, mean, momentum)
        else:
            x = self.do_batch_norm(x, beta, gamma, moving_mean, moving_var)
        return x

    def global_batch_norm0d(self, x, is_training, beta, gamma, momentum,
                            moving_mean, moving_var):
        assert self.ndim(x) == 2
        return self.global_batch_norm(x, is_training, beta, gamma, momentum,
                                      moving_mean, moving_var)

    def global_batch_norm1d(self, x, is_training, beta, gamma, momentum,
                            moving_mean, moving_var):
        assert self.ndim(x) == 3
        return self.global_batch_norm(x, is_training, beta, gamma, momentum,
                                      moving_mean, moving_var)

    def global_batch_norm2d(self, x, is_training, beta, gamma, momentum,
                            moving_mean, moving_var):
        assert self.ndim(x) == 4
        return self.global_batch_norm(x, is_training, beta, gamma, momentum,
                                      moving_mean, moving_var)

    def global_batch_norm3d(self, x, is_training, beta, gamma, momentum,
                            moving_mean, moving_var):
        assert self.ndim(x) == 5
        return self.global_batch_norm(x, is_training, beta, gamma, momentum,
                                      moving_mean, moving_var)
