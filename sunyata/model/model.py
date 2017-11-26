import numpy as np

from .. import backend as Z


class Model(object):
    def __init__(self, spec):
        self.spec = spec
        self.layer = None

    def build(self):
        self.layer, out_form = self.spec.build_one(None)
        self.params = self.layer.get_params()

    def _ensure_built(self):
        if self.layer is None:
            self.build()

    def fit_on_batch(self, x, y, opt, losses, aux_metrics):
        grads_and_params, loss_values, aux_metric_values = Z.gradients(
            self.params, self.layer.forward_multi, losses, aux_metrics,
            [x], [y])
        loss = Z.scalar(loss_values[0])
        acc = Z.scalar(aux_metric_values[0][0])
        assert not np.isnan(loss)
        assert not np.isnan(acc)
        opt.step(grads_and_params)
        return {
            'loss': loss,
            'acc': acc,
        }

    def stats(self, x):
        return {
            'mean': x.mean(),
            'std': x.std(),
        }

    def fit_on_epoch(self, data, opt, losses, aux_metrics, batch_size):
        (x_train, y_train), (x_val, y_val) = data
        batches_per_epoch = len(x_train) // batch_size
        results = []
        for batch_id in range(batches_per_epoch):
            i = batch_id * batch_size
            x = x_train[i:i + batch_size]
            x = Z.constant(Z.numpy_to_device(x))
            y = y_train[i:i + batch_size]
            y = Z.constant(Z.numpy_to_device(y))
            result = self.fit_on_batch(x, y, opt, losses, aux_metrics)
            results.append(result)
        losses = np.array(list(map(lambda x: x['loss'], results)))
        accs = np.array(list(map(lambda x: x['acc'], results)))
        ret = {
            'loss': self.stats(losses),
            'acc': self.stats(accs),
        }
        print(ret)
        return ret

    def fit(self, data, opt, losses, aux_metrics, num_epochs, batch_size):
        self._ensure_built()
        history = []
        for epoch_id in range(num_epochs):
            print('Epoch %d:' % epoch_id)
            epoch_info = self.fit_on_epoch(
                data, opt, losses, aux_metrics, batch_size)
            history.append(epoch_info)
        return history
