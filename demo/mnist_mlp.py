import keras
import numpy as np

from sunyata import backend as Z
from sunyata.metric import *  # noqa
from sunyata.node import *  # noqa


class Optimizer(object):
    def set_params(self, params):
        self.params = params

    def update_param(self, gradient, param):
        raise NotImplementedError

    def update(self, grads_and_params):
        for grad, param in grads_and_params:
            self.update_param(grad, param)


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def update_param(self, gradient, variable):
        Z.assign(variable, Z.variable_to_tensor(variable) - self.lr * gradient)


def one_hot(indices, num_classes, dtype):
    assert indices.ndim == 1
    assert isinstance(num_classes, int)
    assert 0 < num_classes
    assert dtype in Z.supported_dtypes()
    x = np.zeros((len(indices), num_classes), dtype)
    x[np.arange(len(indices)), indices] = 1
    return x


def scale_pixels(x):
    return (x / 255 - 0.5) * 2


def get_data(dtype):
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, 1).astype(dtype)
    x_train = scale_pixels(x_train)
    y_train = one_hot(y_train, 10, dtype)
    x_val = np.expand_dims(x_val, 1).astype(dtype)
    x_val = scale_pixels(x_val)
    y_val = one_hot(y_val, 10, dtype)
    return (x_train, y_train), (x_val, y_val)


dtype = Z.default_dtype()
image_shape = 1, 28, 28
hidden_dim = 100
lr = 0.05
num_epochs = 10
batch_size = 64

(x_train, y_train), (x_val, y_val) = get_data(dtype)

num_classes = y_train.shape[1]
batches_per_epoch = len(x_train) // batch_size

model = SequenceSpec([
    InputSpec(image_shape, dtype),
    FlattenSpec(),
    DenseSpec(hidden_dim),
    ReLUSpec(),
    DenseSpec(num_classes),
    SoftmaxSpec(),
])
model, out_shape = model.build_one(None)

opt = SGD(lr)
opt.set_params(model.params())

cxe = CategoricalCrossEntropy()
acc = CategoricalAccuracy()
judges = [cxe]
aux_judges = [[acc]]

for epoch_id in range(num_epochs):
    for batch_id in range(batches_per_epoch):
        i = batch_id * batch_size
        x = x_train[i:i + batch_size]
        x = Z.constant(Z.cast_numpy_to(x))
        y = y_train[i:i + batch_size]
        y = Z.constant(Z.cast_numpy_to(y))
        grads_and_params, losses, aux_metrics = Z.gradients(
            opt.params, model.forward_multi, judges, aux_judges, [x], [y])
        loss = Z.scalar(losses[0])
        acc = Z.scalar(aux_metrics[0][0])
        assert not np.isnan(loss)
        assert not np.isnan(acc)
        print('epoch %4d batch %4d loss %.4f acc %.4f' %
              (epoch_id, batch_id, loss, acc))
        opt.update(grads_and_params)
