import mxnet as mx
import numpy as np

from ...base.core.variable import BaseVariableAPI


class MXNetVariableAPI(BaseVariableAPI):
    def __init__(self):
        BaseVariableAPI.__init__(self)

    def constant(self, x):
        return x.copy()

    def variable(self, x):
        x = x.copy()
        x.attach_grad()
        return x

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        scores = []
        score_grads = []
        with mx.autograd.record():
            yy_pred = forward(xx, True)
            for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
                scores.append(self.mean(judge(y_true, y_pred)))
                arr = np.ones((1,), self.dtype_of(y_true)) * judge.importance
                score_grads.append(self.numpy_to_device(arr))
        mx.autograd.backward(scores, score_grads)
        grads_and_params = list(map(lambda x: (x.grad, x), params))
        aux_scores = self._aux_scores(aux_judges, yy_true, yy_pred)
        return grads_and_params, scores, aux_scores

    def variable_to_tensor(self, x):
        return x

    def result_to_tensor(self, x):
        return x

    def to_tensor(self, x):
        return x

    def assign(self, x, value):
        x[:] = value
        if x.grad is not None:
            x.grad[:] = 0

    def assign_momentum(self, x, value, momentum):
        if x.grad is None:
            with mx.autograd.pause():
                self.assign(x, momentum * x + (1 - momentum) * value)
        else:
            self.assign(x, momentum * x + (1 - momentum) * value)

    def numpy(self, x):
        return x.asnumpy()
