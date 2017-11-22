from sunyata.dataset.cifar import load_cifar
from sunyata.metric import *  # noqa
from sunyata.model import Model
from sunyata.node import *  # noqa
from sunyata.optim import *  # noqa


def layer(dim):
    return SequenceSpec([
        DenseSpec(dim),
        GlobalBatchNormSpec(),
        ReLUSpec(),
        DropoutSpec(),
    ])


train, val, class_names = load_cifar(classes=20)
print(train[0].shape, train[1].shape)
data = train, val

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
    layer(512),
    layer(512),
    layer(512),
    DenseSpec(num_classes),
    SoftmaxSpec(),
])

opt = NAG()

losses = [CategoricalCrossEntropy()]
aux_metrics = [[CategoricalAccuracy()]]

model = Model(spec)
model.fit(data, opt, losses, aux_metrics, num_epochs, batch_size)
