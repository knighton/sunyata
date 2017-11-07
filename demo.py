import torch
from torch.autograd import Variable


class Layer(object):
    def params(self):
        return []

    def forward(self, x):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, kernel):
        self.kernel = kernel

    def params(self):
        return [self.kernel]

    def forward(self, x):
        return x.mm(self.kernel)


class ReLU(Layer):
    def forward(self, x):
        return x.clamp(min=0)


class Sequence(Layer):
    def __init__(self, layers):
        self.layers = layers

    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params()
        return params

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


def mean_squared_error(true, pred):
    return (true - pred).pow(2).sum()


class Optimizer(object):
    def set_params(self, params):
        self.params = params

    def update_param(self, param):
        raise NotImplementedError

    def update(self):
        for param in self.params:
            self.update_param(param)


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def update_param(self, param):
        param.data -= self.lr * param.grad.data
        param.grad.data.zero_()


dtype = torch.FloatTensor

batch_size, in_dim, hidden_dim, num_classes = 64, 1000, 100, 10
lr = 1e-6

x = Variable(torch.randn(batch_size, in_dim).type(dtype), requires_grad=False)
y = Variable(torch.randn(batch_size, num_classes).type(dtype),
             requires_grad=False)

w1 = Variable(torch.randn(in_dim, hidden_dim).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(hidden_dim, num_classes).type(dtype),
              requires_grad=True)

model = Sequence([
    Dense(w1),
    ReLU(),
    Dense(w2),
])

opt = SGD(lr)
opt.set_params(model.params())

for t in range(500):
    y_pred = model.forward(x)

    loss = mean_squared_error(y, y_pred)
    print(t, loss.data[0])

    loss.backward()

    opt.update()
