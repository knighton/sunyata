from sunyata.dataset.cifar import load_cifar
from sunyata.metric import *  # noqa
from sunyata.model import Model
from sunyata.node import *  # noqa
from sunyata.optim import *  # noqa


train, val, class_names = load_cifar(classes=100)
print(train[0].shape, train[1].shape)
data = train, val

first_hidden_dim = 256
second_hidden_dim = 64
num_epochs = 50
batch_size = 64

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
