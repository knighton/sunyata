import numpy as np
import torch
from torch import _TensorBase
from torch.autograd import Variable

from ...base.core.variable import BaseVariableAPI


class PyTorchVariableAPI(BaseVariableAPI):
    def constant(self, x):
        return Variable(x.clone(), requires_grad=False)

    def variable(self, x):
        return Variable(x.clone(), requires_grad=True)

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        yy_pred = forward(xx, True)
        score_vars = []
        score_grads = []
        for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
            score_vars.append(self.mean(judge(y_true, y_pred)))
            arr = np.ones((1,), self.dtype_of(y_true)) * judge.importance
            score_grads.append(self.numpy_to_device(arr))
        torch.autograd.backward(score_vars, score_grads)
        scores = list(map(lambda x: x.data, score_vars))
        grads_and_params = []
        for param in params:
            grad = None if param.grad is None else param.grad.data
            grads_and_params.append((grad, param))
        aux_scores = self._aux_scores(aux_judges, yy_true, yy_pred)
        return grads_and_params, scores, aux_scores

    def variable_to_tensor(self, x):
        return x.data

    def result_to_tensor(self, x):
        return x.data

    def to_tensor(self, x):
        if isinstance(x, Variable):
            x = x.data
        elif isinstance(x, _TensorBase):
            pass
        else:
            assert False
        return x

    def assign(self, x, value):
        x.data = value
        if x.grad is not None:
            x.grad.data.zero_()

    def numpy(self, x):
        if isinstance(x, torch._TensorBase):
            pass
        elif isinstance(x, Variable):
            x = x.data
        else:
            assert False
        return x.cpu().numpy()
