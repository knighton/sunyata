from sunyata.dataset.cifar import load_cifar
from sunyata.metric import *  # noqa
from sunyata.model import Model
from sunyata.node import *  # noqa
from sunyata.optim import *  # noqa


def make_dense(image_shape, dtype, num_classes):
    def layer(dim):
        return SequenceSpec([
            DenseSpec(dim),
            GlobalBatchNormSpec(),
            ReLUSpec(),
            DropoutSpec(),
        ])

    return SequenceSpec([
        InputSpec(image_shape, dtype),
        FlattenSpec(),
        layer(512),
        layer(512),
        layer(512),
        DenseSpec(num_classes),
        SoftmaxSpec(),
    ])


def make_conv(image_shape, dtype, num_classes):
    def layer(dim):
        return SequenceSpec([
            ConvSpec(dim),
            GlobalBatchNormSpec(),
            ReLUSpec(),
            MaxPoolSpec(),
        ])

    return SequenceSpec([
        InputSpec(image_shape, dtype),
        layer(8),
        layer(8),
        layer(8),
        layer(8),
        FlattenSpec(),
        DenseSpec(num_classes),
        SoftmaxSpec(),
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

spec = make_conv(image_shape, dtype, num_classes)

opt = NAG()

losses = [CategoricalCrossEntropy()]
aux_metrics = [[CategoricalAccuracy()]]

model = Model(spec)
model.fit(data, opt, losses, aux_metrics, num_epochs, batch_size)
