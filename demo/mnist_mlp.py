from sunyata.dataset.mnist import load_mnist
from sunyata.metric import *  # noqa
from sunyata.net import *  # noqa
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

opt = NAG()

losses = [CategoricalCrossEntropy()]
aux_metrics = [[CategoricalAccuracy()]]

stage_1 = Dense(first_hidden_dim) > GlobalBatchNorm > ReLU > Dropout
stage_2 = Dense(second_hidden_dim) > GlobalBatchNorm > ReLU > Dropout

model = Input(image_shape, dtype) > Flatten > stage_1 > stage_2 > \
    Dense(num_classes) > Softmax

model.fit(data, opt, losses, aux_metrics, num_epochs, batch_size)
