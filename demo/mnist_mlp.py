import keras
import numpy as np

from sunyata import backend as Z
from sunyata.dataset.mnist import load_mnist
from sunyata.metric import *  # noqa
from sunyata.node import *  # noqa
from sunyata.optim import *  # noqa


dtype = Z.default_dtype()
image_shape = 1, 28, 28
hidden_dim = 100
lr = 0.05
num_epochs = 10
batch_size = 64

(x_train, y_train), (x_val, y_val) = load_mnist(dtype)

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
