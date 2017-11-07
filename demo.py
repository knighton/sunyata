import numpy as np
import torch
from torch.autograd import Variable


class API(object):
    def dtype(self, x):
        assert isinstance(x, torch.FloatTensor)
        return np.float32


Z = API()


class Form(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def check(self, x):
        assert tuple(x.size()[1:]) == self.shape
        assert Z.dtype(x) == self.dtype


class Layer(object):
    def params(self):
        return []

    def forward(self, x):
        raise NotImplementedError


class InputLayer(Layer):
    def __init__(self, form):
        self.form = form

    def forward(self, x):
        self.form.check(x)
        return x


class DenseLayer(Layer):
    def __init__(self, kernel, bias):
        self.kernel = Variable(torch.FloatTensor(kernel), requires_grad=True)
        self.bias = Variable(torch.FloatTensor(bias), requires_grad=True)

    def params(self):
        return [self.kernel, self.bias]

    def forward(self, x):
        return x.mm(self.kernel) + self.bias


class ReLULayer(Layer):
    def forward(self, x):
        return x.clamp(min=0)


class SequenceLayer(Layer):
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


class Spec(object):
    def build(self, form=None):
        raise NotImplementedError


class InputSpec(Spec):
    def __init__(self, shape, dtype):
        self.form = Form(shape, dtype)

    def build(self, form=None):
        assert form is None
        return InputLayer(self.form), self.form


class DenseSpec(Spec):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def build(self, form=None):
        in_dim, = form.shape
        kernel = np.random.normal(0, 1, (in_dim, self.out_dim))
        bias = np.random.normal(0, 1, (self.out_dim,))
        out_shape = self.out_dim,
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)


class ReLUSpec(Spec):
    def build(self, form=None):
        return ReLULayer(), form


class SequenceSpec(Spec):
    def __init__(self, specs):
        self.specs = specs

    def build(self, form=None):
        layers = []
        for spec in self.specs:
            layer, form = spec.build(form)
            layers.append(layer)
        return SequenceLayer(layers), form


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

model = SequenceSpec([
    InputSpec((in_dim,), np.float32),
    DenseSpec(hidden_dim),
    ReLUSpec(),
    DenseSpec(num_classes),
])

model, out_shape = model.build()

opt = SGD(lr)
opt.set_params(model.params())

for t in range(500):
    y_pred = model.forward(x)

    loss = mean_squared_error(y, y_pred)
    print(t, loss.data[0])

    loss.backward()

    opt.update()
