import keras
import numpy as np

from sunyata import backend as Z
from sunyata.dataset.mnist import load_mnist
from sunyata.metric import *  # noqa
from sunyata.model import Model
from sunyata.node import *  # noqa
from sunyata.optim import *  # noqa


dtype = Z.default_dtype()
image_shape = 1, 28, 28
hidden_dim = 100
lr = 0.05
num_epochs = 10
batch_size = 64

data = load_mnist(dtype)

num_classes = data[0][1].shape[1]

spec = SequenceSpec([
    InputSpec(image_shape, dtype),
    FlattenSpec(),
    DenseSpec(hidden_dim),
    ReLUSpec(),
    DenseSpec(num_classes),
    SoftmaxSpec(),
])

opt = SGD(lr)

losses = [CategoricalCrossEntropy()]
aux_metrics = [[CategoricalAccuracy()]]

model = Model(spec)
model.fit(data, opt, losses, aux_metrics, num_epochs, batch_size)
