class OptimizerContext(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Optimizer(object):
    def __init__(self):
        self.vid2context = {}

    def make_context(self, variable):
        raise NotImplementedError

    def learn(self, context, gradients, variable):
        raise NotImplementedError

    def step(self, grads_and_vars):
        for gradient, variable in grads_and_vars:
            variable_id = id(variable)
            context = self.vid2context.get(variable_id)
            if context is None:
                context = self.make_context(variable)
                self.vid2context[variable_id] = context
            self.learn(variable, gradient, context)
