import tensorflow as tf
import tensorflow.contrib.eager as tfe

from ...base.core.variable import BaseVariableAPI


class TensorFlowVariableAPI(BaseVariableAPI):
    def __init__(self):
        BaseVariableAPI.__init__(self)

    def constant(self, x):
        return tf.constant(x)

    def variable(self, x):
        return tfe.Variable(x, name=self._variable_name())

    def _ivag_inner(self, forward, judges, aux_judges, xx, yy_true, bridge):
        yy_pred = forward(xx)
        scores = []
        for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
            scores.append(self.mean(judge(y_true, y_pred)) * judge.importance)
        bridge.append(self._aux_scores(aux_judges, yy_true, yy_pred))
        return scores

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        ivag = tfe.implicit_value_and_gradients(self._ivag_inner)
        bridge = []
        scores, grads_and_params = \
            ivag(forward, judges, aux_judges, xx, yy_true, bridge)
        aux_scores = bridge.pop()
        return grads_and_params, scores, aux_scores

    def variable_to_tensor(self, x):
        return x[:]

    def result_to_tensor(self, x):
        return x

    def assign(self, x, value):
        x.assign(value)

    def numpy(self, x):
        return x.numpy()
