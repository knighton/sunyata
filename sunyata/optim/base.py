class Optimizer(object):
    def set_params(self, params):
        self.params = params

    def update_param(self, gradient, param):
        raise NotImplementedError

    def update(self, grads_and_params):
        for grad, param in grads_and_params:
            self.update_param(grad, param)
