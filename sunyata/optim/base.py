class OptimizerContext(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Optimizer(object):
    def __init__(self):
        self.vid2context = {}

    def make_context(self, variable):
        raise NotImplementedError

    def update_variable(self, var, grad, ctx):
        raise NotImplementedError

    def step(self, grads_and_vars):
        for gradient, variable in grads_and_vars:
            variable_id = id(variable)
            context = self.vid2context.get(variable_id)
            if context is None:
                context = OptimizerContext(**self.make_context(variable))
                self.vid2context[variable_id] = context
            if gradient is None:
                continue
            self.update_variable(variable, gradient, context)
