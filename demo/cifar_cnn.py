from sunyata.dataset.cifar import load_cifar
from sunyata.metric import *  # noqa
from sunyata.net import *  # noqa
from sunyata.optim import *  # noqa


def make_dense(image_shape, dtype, num_classes):
    return Input(image_shape, dtype) > Flatten > \
        (Dense(256) > GlobalBatchNorm > ReLU > Dropout) * 3 > \
        Dense(num_classes) > Softmax


def make_conv(image_shape, dtype, num_classes):
    return Input(image_shape, dtype) > \
        (Conv(16) > GlobalBatchNorm > ReLU > MaxPool) * 4 > \
        Flatten > Dense(num_classes) > Softmax


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

model = make_conv(image_shape, dtype, num_classes)

opt = NAG()

losses = [CategoricalCrossEntropy()]
aux_metrics = [[CategoricalAccuracy()]]

model.fit(data, opt, losses, aux_metrics, num_epochs, batch_size)
