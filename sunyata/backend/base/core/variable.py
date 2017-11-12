from ..base import APIBase


class BaseVariableAPI(APIBase):
    def constant(self, x):
        raise NotImplementedError

    def _variable_name(self, name=None):
        if name is None:
            name = str(1 << 30)
        else:
            assert isinstance(name, str)
            assert name
        return name

    def variable(self, x):
        raise NotImplementedError

    def _aux_scores(self, aux_judges, yy_true, yy_pred):
        if aux_judges is None:
            return None
        aux_scores = []
        for y_aux_judges, y_true, y_pred in zip(aux_judges, yy_true, yy_pred):
            y_aux_scores = []
            for judge in y_aux_judges:
                result = self.mean(judge(y_true, y_pred))
                y_aux_scores.append(self.result_to_tensor(result))
            aux_scores.append(y_aux_scores)
        return aux_scores

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        raise NotImplementedError

    def variable_to_tensor(self, x):
        raise NotImplementedError

    def result_to_tensor(self, x):
        raise NotImplementedError

    def assign(self, x, new_value):
        raise NotImplementedError

    def incr(self, x, incr):
        self.assign(x, self.variable_to_tensor(x) + incr)

    def decr(self, x, decr):
        self.assign(x, self.variable_to_tensor(x) - decr)

    def numpy(self, x):
        raise NotImplementedError

    def list(self, x):
        return self.numpy(x).tolist()

    def scalar(self, x):
        assert self.size(x) == 1
        return self.numpy(x).flatten()[0]
