from sunyata.dataset.mnist import load_mnist
from sunyata.metric import *  # noqa
from sunyata.model import Model
from sunyata.node import *  # noqa
from sunyata.optim import *  # noqa


first_hidden_dim = 256
second_hidden_dim = 64
num_epochs = 10
batch_size = 64

data = load_mnist()
x = data[0][0][0]
dtype = x.dtype.name
image_shape = x.shape
y = data[0][1][0]
num_classes = len(y)

spec = SequenceSpec([
    InputSpec(image_shape, dtype),
    FlattenSpec(),
    DenseSpec(first_hidden_dim),
    GlobalBatchNormSpec(),
    ReLUSpec(),
    DropoutSpec(),
    DenseSpec(second_hidden_dim),
    GlobalBatchNormSpec(),
    ReLUSpec(),
    DropoutSpec(),
    DenseSpec(num_classes),
    SoftmaxSpec(),
])

opt = NAG()

losses = [CategoricalCrossEntropy()]
aux_metrics = [[CategoricalAccuracy()]]

model = Model(spec)
model.fit(data, opt, losses, aux_metrics, num_epochs, batch_size)
