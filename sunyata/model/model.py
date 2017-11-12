import numpy as np

from .. import backend as Z


class Model(object):
    def __init__(self, spec):
        self.spec = spec
        self.layer = None

    def build(self):
        self.layer, out_shape = self.spec.build_one(None)

    def _ensure_built(self):
        if self.layer is None:
            self.build()

    def fit(self, data, opt, losses, aux_metrics, num_epochs, batch_size):
        self._ensure_built()
        opt.set_params(self.layer.params())
        (x_train, y_train), (x_val, y_val) = data
        batches_per_epoch = len(x_train) // batch_size
        for epoch_id in range(num_epochs):
            for batch_id in range(batches_per_epoch):
                i = batch_id * batch_size
                x = x_train[i:i + batch_size]
                x = Z.constant(Z.cast_numpy_to(x))
                y = y_train[i:i + batch_size]
                y = Z.constant(Z.cast_numpy_to(y))
                grads_and_params, loss_values, aux_metric_values = Z.gradients(
                    opt.params, self.layer.forward_multi, losses, aux_metrics,
                    [x], [y])
                loss = Z.scalar(loss_values[0])
                acc = Z.scalar(aux_metric_values[0][0])
                assert not np.isnan(loss)
                assert not np.isnan(acc)
                print('epoch %4d batch %4d loss %.4f acc %.4f' %
                      (epoch_id, batch_id, loss, acc))
                opt.update(grads_and_params)
