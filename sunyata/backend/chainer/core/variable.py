import chainer
from chainer import Variable
import numpy as np

from ...base.core.variable import BaseVariableAPI


class ChainerVariableAPI(BaseVariableAPI):
    def __init__(self):
        BaseVariableAPI.__init__(self)

    def constant(self, x):
        return Variable(x, requires_grad=False)

    def variable(self, x):
        return Variable(x)

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        yy_pred = forward(xx, True)
        score_vars = []
        score_grads = []
        for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
            score_vars.append(self.mean(judge(y_true, y_pred)))
            arr = np.ones((1,), self.dtype_of(y_true)) * judge.importance
            score_grads.append(self.numpy_to_device(arr))
        grad_vars = chainer.grad(score_vars, params, score_grads)
        scores = list(map(lambda x: x.data, score_vars))
        grads_and_params = []
        for param, grad_var in zip(params, grad_vars):
            grads_and_params.append((grad_var.data, param))
        aux_scores = self._aux_scores(aux_judges, yy_true, yy_pred)
        return grads_and_params, scores, aux_scores

    def variable_to_tensor(self, x):
        return x.data

    def result_to_tensor(self, x):
        return x.data

    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            pass
        elif isinstance(x, Variable):
            x = x.data
        else:
            assert False
        return x

    def assign(self, x, value):
        x.data = value.copy()

    def assign_momentum(self, x, value, momentum):
        self.assign(x, momentum * self.variable_to_tensor(x) +
                    (1 - momentum) * self.to_tensor(value))

    def numpy(self, x):
        return x.data.copy() if isinstance(x, Variable) else x.copy()
